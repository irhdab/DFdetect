import os
import shutil
import tempfile
import uuid
import random
import asyncio
import base64
from pathlib import Path
import time
import socket
import math

import uvicorn
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form, WebSocket, Query
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketDisconnect

# Create FastAPI app
app = FastAPI(
    title="Deepfake Detection API",
    description="API for real-time deepfake detection on videos",
    version="1.0.0"
)

# Set a higher file size limit for uploads (100MB)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if they don't exist
Path("app/uploads").mkdir(parents=True, exist_ok=True)
Path("app/models/weights").mkdir(parents=True, exist_ok=True)

# Set up templates and static files
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Define available models
AVAILABLE_MODELS = ["mesonet", "xceptionnet"]

# Store active websocket connections
active_connections = []

class ProcessingResult(BaseModel):
    """Model for video processing results"""
    frame: int
    confidence_fake: float
    overlay_frame: Optional[str] = None

class VideoInfo(BaseModel):
    """Model for video information"""
    total_frames: int
    fps: float
    width: int
    height: int

class VideoResult(BaseModel):
    """Model for complete video results"""
    results: List[ProcessingResult]
    video_info: VideoInfo

def get_model_info(model_name):
    """Get model information without requiring TensorFlow"""
    model_name = model_name.lower()
    
    if model_name == "mesonet":
        return {
            "id": "mesonet",
            "name": "MesoNet",
            "description": "Lightweight CNN (faster)"
        }
    elif model_name == "xceptionnet":
        return {
            "id": "xceptionnet",
            "name": "XceptionNet",
            "description": "Deep CNN (more accurate)"
        }
    else:
        return {
            "id": "unknown",
            "name": "Unknown Model",
            "description": "Unknown model type"
        }

def get_available_models():
    """Get list of available models"""
    return [get_model_info(model) for model in AVAILABLE_MODELS]

def generate_mock_results(model_name, frame_count=30, frame_interval=5):
    """Generate mock analysis results with multiple frames"""
    results = []
    accuracy_boost = 0.1 if model_name == "xceptionnet" else 0.0
    total_frames = frame_count * frame_interval
    
    # Generate a sequence of frames at regular intervals
    for i in range(0, total_frames, frame_interval):
        frame_num = i
        # Generate confidence score based on frame number with some randomness
        confidence = min(0.3 + (i / total_frames * 0.7) + random.uniform(-0.2, 0.2) + accuracy_boost, 1.0)
        results.append({"frame": frame_num, "confidence_fake": confidence})
    
    return results

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main page"""
    return templates.TemplateResponse("index.html", {"request": request, "models": get_available_models()})

@app.get("/api/models")
async def get_models():
    """Get available models"""
    return get_available_models()

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...), model: str = Form("mesonet")):
    """Upload and process a video file"""
    # Validate model
    if model not in AVAILABLE_MODELS:
        model = "mesonet"  # Default to MesoNet if invalid
        
    # Generate a unique filename
    file_id = str(uuid.uuid4())
    
    # Return immediate response with job ID and selected model
    return {
        "file_id": file_id,
        "model": model,
        "message": f"Video upload successful, processing with {model} started",
        "status_endpoint": f"/status/{file_id}"
    }

@app.get("/status/{file_id}")
async def get_status(file_id: str):
    """Get the status of video processing"""
    return {"status": "completed", "result_endpoint": f"/result/{file_id}"}

@app.get("/result/{file_id}")
async def get_result(file_id: str):
    """Get the results of video processing"""
    # Generate a complete set of results across the entire video
    total_frames = 600  # Mock value for demo
    fps = 30
    frame_interval = 5  # Analyze every 5th frame
    
    # Generate a random but reasonable duration (between 5-30 seconds)
    duration = random.uniform(5, 30)
    
    # Determine which model was used (in a real app, this would be stored)
    # For demo purposes, we'll randomize it
    model = random.choice(AVAILABLE_MODELS)
    model_info = get_model_info(model)
    accuracy_boost = 0.05 if model == "xceptionnet" else 0.0
    
    # Create a unique pattern for this video based on a hash of the file_id
    id_hash = hash(file_id) % 10000
    rng = random.Random(id_hash)
    
    pattern_type = rng.randint(0, 9)
    base_confidence = 0.2 + (pattern_type / 10.0) * 0.6
    pattern_shape = rng.randint(0, 3)
    results = []
    max_frame = min(total_frames, int(duration * fps))
    
    for frame_idx in range(0, max_frame, frame_interval):
        if frame_idx / fps > duration:
            break
        position = frame_idx / max_frame
        if pattern_shape == 0:
            variation = rng.uniform(-0.15, 0.15)
            confidence = base_confidence + variation
        elif pattern_shape == 1:
            frequency = rng.uniform(5, 20)
            amplitude = rng.uniform(0.1, 0.3)
            confidence = base_confidence + amplitude * math.sin(position * frequency)
        elif pattern_shape == 2:
            peak_position = rng.uniform(0.3, 0.7)
            spread = rng.uniform(0.1, 0.3)
            height = rng.uniform(0.1, 0.4)
            confidence = base_confidence + height * math.exp(-((position - peak_position) ** 2) / (2 * spread ** 2))
        else:
            shift_direction = rng.choice([-1, 1])
            shift_amount = rng.uniform(0.1, 0.3)
            confidence = base_confidence + shift_direction * shift_amount * position
        
        noise = rng.uniform(-0.05, 0.05)
        confidence += noise
        confidence = min(max(confidence + accuracy_boost, 0.05), 0.95)
        
        results.append({
            "frame": frame_idx,
            "confidence_fake": round(confidence, 2)
        })
    
    return {
        "model": model_info,
        "results": results,
        "video_info": {
            "total_frames": total_frames,
            "fps": fps,
            "width": 1280,
            "height": 720,
            "duration": duration
        }
    }

@app.post("/process/")
async def process_video(file: UploadFile = File(...), model: str = Form("mesonet")):
    """Process a video file and return results"""
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate model
    if model not in AVAILABLE_MODELS:
        model = "mesonet"  # Default to MesoNet if invalid
    
    # Get model info
    model_info = get_model_info(model)
    
    # Save file to a temporary location to get video properties
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    try:
        contents = await file.read()
        with open(temp_file.name, 'wb') as f:
            f.write(contents)
            
        # Try to get actual video properties using OpenCV
        try:
            import cv2
            cap = cv2.VideoCapture(temp_file.name)
            if cap.isOpened():
                # Get video info
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                # Duration in seconds
                duration = total_frames / fps if fps > 0 else 0
                
                # Fall back to defaults if values are unreasonable
                if total_frames <= 0 or fps <= 0:
                    total_frames = 600
                    fps = 30
                    duration = 20
            else:
                # Fallback values
                total_frames = 600
                fps = 30
                width = 1280
                height = 720
                duration = 20
                
            cap.release()
        except ImportError:
            # Fallback if OpenCV is not available
            total_frames = 600
            fps = 30
            width = 1280
            height = 720
            duration = 20
        
        # Generate more realistic results with proper frame distribution
        frame_interval = 5  # Process every 5th frame
        mock_results = []
        accuracy_boost = 0.05 if model == "xceptionnet" else 0.0
        
        # Create a unique pattern for this video based on a hash of the filename
        filename_hash = hash(file.filename if file.filename else "default") % 10000
        rng = random.Random(filename_hash)
        
        pattern_type = rng.randint(0, 9)
        base_confidence = 0.2 + (pattern_type / 10.0) * 0.6
        pattern_shape = rng.randint(0, 3)
        
        for frame_idx in range(0, total_frames, frame_interval):
            if frame_idx / fps > duration:
                break
                
            position = frame_idx / total_frames
            if pattern_shape == 0:
                variation = rng.uniform(-0.15, 0.15)
                confidence = base_confidence + variation
            elif pattern_shape == 1:
                frequency = rng.uniform(5, 20)
                amplitude = rng.uniform(0.1, 0.3)
                confidence = base_confidence + amplitude * math.sin(position * frequency)
            elif pattern_shape == 2:
                peak_position = rng.uniform(0.3, 0.7)
                spread = rng.uniform(0.1, 0.3)
                height = rng.uniform(0.1, 0.4)
                confidence = base_confidence + height * math.exp(-((position - peak_position) ** 2) / (2 * spread ** 2))
            else:
                shift_direction = rng.choice([-1, 1])
                shift_amount = rng.uniform(0.1, 0.3)
                confidence = base_confidence + shift_direction * shift_amount * position
            
            noise = rng.uniform(-0.05, 0.05)
            confidence += noise
            confidence = min(max(confidence + accuracy_boost, 0.05), 0.95)
            
            mock_results.append({
                "frame": frame_idx,
                "confidence_fake": round(confidence, 2)
            })
        
        return {
            "model": model_info,
            "results": mock_results,
            "video_info": {
                "total_frames": total_frames,
                "fps": fps,
                "width": width,
                "height": height,
                "duration": duration
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        # Clean up temporary file
        os.unlink(temp_file.name)

@app.websocket("/ws/webcam")
async def websocket_endpoint(websocket: WebSocket, model: str = Query("mesonet")):
    """WebSocket endpoint for real-time webcam processing"""
    # Validate model
    if model not in AVAILABLE_MODELS:
        model = "mesonet"  # Default to MesoNet if invalid
    
    print(f"New WebSocket connection for webcam processing with model: {model}")
    
    try:
        await websocket.accept()
        active_connections.append(websocket)
        print(f"WebSocket connection accepted, total active: {len(active_connections)}")
        
        # Get model info
        model_info = get_model_info(model)
        accuracy_boost = 0.1 if model == "xceptionnet" else 0.0
        
        # Create a more realistic confidence pattern that evolves over time
        # This simulates a deepfake becoming more detectable as the session continues
        confidence_baseline = 0.3  # Starting baseline
        confidence_trend = 0.001   # Small increase per frame
        confidence_max = 0.85      # Maximum confidence
        
        # Set initial random seed for this session
        random_seed = random.randint(0, 1000)
        random.seed(random_seed)
        
        frame_idx = 0
        last_message_time = time.time()
        message_interval = 0.2  # Send update every 200ms
        
        while True:
            # Add delay between frames for realistic processing
            current_time = time.time()
            if current_time - last_message_time < message_interval:
                await asyncio.sleep(message_interval - (current_time - last_message_time))
            
            # Calculate confidence with evolving pattern
            random.seed(random_seed + frame_idx)  # Deterministic randomness based on frame
            
            # Calculate base confidence that increases slightly over time
            base_confidence = min(confidence_baseline + (frame_idx * confidence_trend), confidence_max)
            
            # Add randomness
            random_factor = random.uniform(-0.15, 0.15)
            
            # Calculate final confidence
            confidence = min(max(base_confidence + random_factor + accuracy_boost, 0.05), 0.95)
            
            # Create result
            result = {
                "frame": frame_idx,
                "confidence_fake": round(confidence, 2),
                "model": model,
                "timestamp": time.time()
            }
            
            # Send result to client
            try:
                await websocket.send_json(result)
                last_message_time = time.time()
                frame_idx += 1
            except Exception as e:
                print(f"Error sending WebSocket message: {e}")
                break
                
            # Allow other tasks to run
            await asyncio.sleep(0.01)
            
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
            print(f"WebSocket removed from active connections, remaining: {len(active_connections)}")

if __name__ == "__main__":
    print("Starting Deepfake Detection Web Interface...")
    print("This is a simplified version without model loading")
    print("Available models:", ", ".join(AVAILABLE_MODELS))
    
    # Check if port 8000 is already in use
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 8000
    
    # Try to bind to port 8000, if it fails try other ports up to 8010
    for test_port in range(8000, 8011):
        try:
            sock.bind(('0.0.0.0', test_port))
            port = test_port
            break
        except socket.error:
            print(f"Port {test_port} is already in use, trying next port...")
            continue
    
    # Close the socket
    sock.close()
    
    print(f"Access the application at http://localhost:{port}")
    
    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port,
            timeout_keep_alive=120,  # Increase keep-alive timeout
            limit_concurrency=10,    # Limit concurrent connections
            timeout_graceful_shutdown=30  # Allow time for graceful shutdown
        )
    except Exception as e:
        print(f"Error starting server: {e}") 