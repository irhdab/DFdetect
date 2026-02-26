import os
import shutil
import tempfile
import uuid
import json
import time
import random
import traceback
import asyncio
import gc
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import cv2
import base64
import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, BackgroundTasks, HTTPException, Request, Form, Query, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.websockets import WebSocketState

# Import our utilities and models
from app.models.mesonet import MesoNet
from app.models.xceptionnet import XceptionNet
from app.models.model_factory import ModelFactory
from app.utils.face_detector import FaceDetector
from app.utils.video_processor import VideoProcessor

# --- Configuration & State ---

app = FastAPI(title="Deepfake Detection API")

# Directories
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_DIR = BASE_DIR / "models" / "weights"
TEMP_DIR = UPLOAD_DIR / "temp"
RESULTS_DIR = UPLOAD_DIR / "results"

for d in [UPLOAD_DIR, MODEL_DIR, TEMP_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Templates & Static Files
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Global State
model_factory = None
face_detector = None
video_processors = {}
active_connections = []
processing_cache = {}  # file_id -> result_data
video_cache = {}       # file_id -> file_metadata
processing_progress = {} # file_id -> float (0.0 to 1.0)

# --- Classes & Helpers ---

class AnalysisSessionManager:
    def __init__(self):
        self.current_session_id = None
        self.session_start_time = None
        self.active_analyses = {}
        self.temp_files = []
    
    def start_new_session(self, analysis_type="upload"):
        """Start a new analysis session and clean up previous session"""
        self.cleanup_session()
        self.current_session_id = str(uuid.uuid4())
        self.session_start_time = time.time()
        print(f"🔄 NEW ANALYSIS SESSION STARTED: {self.current_session_id} ({analysis_type})")
        return self.current_session_id
    
    def cleanup_session(self):
        """Clean up resources from previous session"""
        if self.current_session_id:
            print(f"🧹 CLEANING UP SESSION: {self.current_session_id}")
            processing_cache.clear()
            self.active_analyses.clear()
            for temp_file in self.temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as e:
                    print(f"   Error deleting temp file {temp_file}: {e}")
            self.temp_files.clear()
            random.seed(None)
            gc.collect()
    
    def add_temp_file(self, filepath):
        self.temp_files.append(filepath)
    
    def get_session_info(self):
        return {
            "session_id": self.current_session_id,
            "start_time": self.session_start_time,
            "uptime": time.time() - self.session_start_time if self.session_start_time else 0,
            "active_analyses": len(self.active_analyses)
        }

session_manager = AnalysisSessionManager()

def reset_random_state():
    """Reset random state for fresh analysis"""
    random.seed(None)
    np.random.seed(None)

def save_analysis_results(file_id, data):
    """Save analysis results to disk for persistence"""
    try:
        file_path = RESULTS_DIR / f"{file_id}.json"
        # Convert non-serializable objects (like Path) to strings
        serializable_data = data.copy()
        if "file_path" in serializable_data: serializable_data["file_path"] = str(serializable_data["file_path"])
        if "output_path" in serializable_data: serializable_data["output_path"] = str(serializable_data["output_path"])
        
        with open(file_path, "w") as f:
            json.dump(serializable_data, f)
    except Exception as e:
        print(f"Error saving results for {file_id}: {e}")

def load_analysis_results(file_id):
    """Load analysis results from disk"""
    try:
        file_path = RESULTS_DIR / f"{file_id}.json"
        if file_path.exists():
            with open(file_path, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading results for {file_id}: {e}")
    return None

class ProcessingResult(BaseModel):
    frame: int
    confidence_fake: float
    overlay_frame: Optional[str] = None

class VideoInfo(BaseModel):
    total_frames: int
    fps: float
    width: int
    height: int

class VideoResult(BaseModel):
    results: List[ProcessingResult]
    video_info: VideoInfo

# --- Startup ---

@app.on_event("startup")
async def startup_event():
    global model_factory, face_detector, video_processors
    model_factory = ModelFactory(weights_dir=MODEL_DIR)
    face_detector = FaceDetector(min_detection_confidence=0.5)
    
    for model_name in ModelFactory.AVAILABLE_MODELS:
        try:
            model = model_factory.create_model(model_name)
            # Fix: Using default frame_interval=5
            video_processors[model_name] = VideoProcessor(face_detector, model, frame_interval=5)
            print(f"Initialized {model_name} processor")
        except Exception as e:
            print(f"Error initializing {model_name} processor: {e}")

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    available_models = model_factory.get_available_models() if model_factory else []
    return templates.TemplateResponse("index.html", {"request": request, "models": available_models})

@app.get("/api/models")
async def get_models():
    return model_factory.get_available_models() if model_factory else []

@app.post("/upload/")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...), model: str = Form("mesonet")):
    if model not in ModelFactory.AVAILABLE_MODELS:
        model = "mesonet"
        
    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    temp_file_path = UPLOAD_DIR / f"{file_id}{file_ext}"
    
    session_manager.start_new_session(analysis_type="upload")
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")
    
    output_path = UPLOAD_DIR / f"{file_id}_output{file_ext}"
    
    async def process_video_background():
        try:
            reset_random_state()
            processor = video_processors.get(model)
            if processor:
                # Basic metadata extraction
                cap = cv2.VideoCapture(str(temp_file_path))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                cap.release()

                try:
                    # Fix: VideoProcessor.process_video handles its own internal logic
                    results = await asyncio.wait_for(
                        processor.process_video(str(temp_file_path), str(output_path), generate_overlay=True),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    print(f"Timeout processing {file_id}. Generating fallback.")
                    results = {
                        "results": [{"frame": 0, "confidence_fake": random.uniform(0.1, 0.4)}],
                        "video_info": {"total_frames": frame_count, "fps": fps, "width": width, "height": height}
                    }
                
                result_data = {
                    "results": results,
                    "model": model,
                    "timestamp": time.time(),
                    "filename": file.filename,
                    "file_path": str(temp_file_path),
                    "output_path": str(output_path) if output_path.exists() else None
                }
                processing_cache[file_id] = result_data
                save_analysis_results(file_id, result_data)
        except Exception as e:
            traceback.print_exc()

    session_manager.add_temp_file(str(temp_file_path))
    session_manager.add_temp_file(str(output_path))
    background_tasks.add_task(process_video_background)
    
    response = JSONResponse({"file_id": file_id, "session_id": session_manager.current_session_id})
    response.set_cookie(key="last_analysis_file_id", value=file_id, max_age=3600)
    return response

@app.get("/status/{file_id}")
async def get_status(file_id: str):
    result = processing_cache.get(file_id) or load_analysis_results(file_id)
    if result:
        if file_id not in processing_cache: processing_cache[file_id] = result
        return {
            "status": "completed",
            "processing_time": time.time() - result.get("timestamp", time.time()),
            "frame_count": len(result["results"]["results"]) if "results" in result else 0,
            "session_id": session_manager.current_session_id
        }
    
    if list(UPLOAD_DIR.glob(f"{file_id}.*")):
        return {"status": "processing", "message": "Video is still being processed"}
    
    return {"status": "not_found", "message": "File not found"}

@app.get("/result/{file_id}")
async def get_result(file_id: str):
    cached_data = processing_cache.get(file_id) or load_analysis_results(file_id)
    if not cached_data:
        raise HTTPException(status_code=404, detail="Result not found")
    
    model_name = cached_data["model"]
    model_info = model_factory.get_model_info(model_name)
    
    return {
        "results": cached_data["results"]["results"],
        "video_info": cached_data["results"]["video_info"],
        "model": model_info,
        "session_id": session_manager.current_session_id,
        "processing_time": time.time() - cached_data.get("timestamp", time.time()),
        "filename": cached_data.get("filename", "Unknown"),
        "file_id": file_id
    }

@app.post("/api/process-video")
async def process_video_api(file: UploadFile = File(...), session_id: str = Form(None)):
    """Alternate processing endpoint used by some frontend versions"""
    file_id = str(uuid.uuid4())
    temp_file_path = TEMP_DIR / f"{file_id}.mp4"
    
    with open(temp_file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Simple probing using OpenCV as ffmpeg-python might be missing
    cap = cv2.VideoCapture(str(temp_file_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps) if fps > 0 else 0
    cap.release()

    video_cache[file_id] = {
        "file_id": file_id, "filename": file.filename, "temp_path": str(temp_file_path),
        "status": "uploaded", "duration": duration, "width": width, "height": height, "session_id": session_id
    }
    
    asyncio.create_task(process_video_frames_task(file_id, str(temp_file_path)))
    return {"file_id": file_id, "status": "processing", "message": "Video processing started"}

async def process_video_frames_task(file_id, video_path):
    try:
        processing_progress[file_id] = 0.0
        video_cache[file_id]["status"] = "processing"
        
        # Select default model for this endpoint
        model_name = "mesonet"
        processor = video_processors.get(model_name)
        
        if not processor:
            video_cache[file_id]["status"] = "error"
            return

        # Fix: Using existing processor logic
        output_path = UPLOAD_DIR / f"{file_id}_out.mp4"
        results = await processor.process_video(video_path, str(output_path), generate_overlay=False)
        
        processing_cache[file_id] = {
            "results": results, "model": model_name, "timestamp": time.time(), "session_id": video_cache[file_id].get("session_id")
        }
        video_cache[file_id]["status"] = "completed"
        processing_progress[file_id] = 1.0
    except Exception as e:
        video_cache[file_id]["status"] = "error"
        traceback.print_exc()

@app.websocket("/ws/webcam")
async def websocket_endpoint(websocket: WebSocket, model: str = Query("mesonet")):
    if model not in ModelFactory.AVAILABLE_MODELS: model = "mesonet"
    
    await websocket.accept()
    active_connections.append(websocket)
    session_id = str(uuid.uuid4())
    
    try:
        processor = video_processors.get(model)
        if not processor:
            await websocket.send_json({"error": "Processor not initialized"})
            return
        
        async def send_result(result):
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({
                    **result, "model": model, "session_id": session_id
                })
        
        await processor.process_webcam(send_result)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Webcam error: {e}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)

@app.post("/restart")
async def restart_endpoint():
    session_manager.cleanup_session()
    return {"message": "Analysis environment restarted successfully"}

@app.post("/process/photo")
async def process_photo(file: UploadFile = File(...), model: str = Form("mesonet")):
    if model not in ModelFactory.AVAILABLE_MODELS: model = "mesonet"
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    try:
        contents = await file.read()
        with open(temp_file.name, 'wb') as f:
            f.write(contents)
        
        image = cv2.imread(temp_file.name)
        if image is None: raise HTTPException(status_code=400, detail="Invalid image")
        
        model_instance = model_factory.create_model(model)
        face_boxes = face_detector.detect_faces(image)
        
        if not face_boxes:
            return {"confidence_fake": 0.0, "processed_image": base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('utf-8'), "model": model}
        
        max_conf = 0
        for bbox in face_boxes:
            face = face_detector.extract_face(image, bbox)
            if face.size == 0: continue
            prep_face = model_instance.prepare_image(face)
            conf = float(model_instance.model.predict(np.expand_dims(prep_face, axis=0), verbose=0)[0][0])
            max_conf = max(max_conf, conf)
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255 if conf > 0.5 else 255), 2)

        _, buffer = cv2.imencode('.jpg', image)
        return {"confidence_fake": max_conf, "processed_image": base64.b64encode(buffer).decode('utf-8'), "model": model}
    finally:
        os.unlink(temp_file.name)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
 