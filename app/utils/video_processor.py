"""
Real VideoProcessor for deepfake detection.

Replaces all mock/random data with actual frame extraction, face detection,
and model inference.
"""

import cv2
import numpy as np
import os
import base64
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path


class VideoProcessor:
    """
    Real video processor that:
    1. Opens a video file with OpenCV
    2. Extracts frames at a configurable interval
    3. Detects faces in each frame using FaceDetector
    4. Runs the deepfake detection model on each detected face
    5. Returns per-frame confidence scores (probability of being fake)
    """

    def __init__(
        self,
        face_detector=None,
        deepfake_model=None,
        frame_interval: int = 15,
        video_path: str = None,
        max_frames: int = 60,
    ):
        """
        Args:
            face_detector: FaceDetector instance
            deepfake_model: MesoNet or XceptionNet instance with a .predict(face_img) method
            frame_interval: Analyze every Nth frame (default 15 = every 0.5s at 30fps)
            video_path: Optional path to video (for extract_frames compatibility)
            max_frames: Cap on number of frames to process per video
        """
        self.face_detector = face_detector
        self.model = deepfake_model
        self.frame_interval = frame_interval
        self.video_path = video_path
        self.max_frames = max_frames
        self.video_info = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_video(
        self,
        video_path: str,
        output_path: str = None,
        generate_overlay: bool = False,
        progress_callback: Callable = None,
    ) -> Dict[str, Any]:
        """
        Process a video file: detect faces per frame and run model inference.

        Args:
            video_path: Path to the input video file
            output_path: (optional) Save annotated video here
            generate_overlay: Include base64-encoded annotated frames in results
            progress_callback: Optional callable(progress: float) [0.0-1.0]

        Returns:
            {
                "results": [{"frame": int, "confidence_fake": float, ...}, ...],
                "video_info": {"total_frames", "fps", "width", "height", "duration"}
            }
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ VideoProcessor: Cannot open video: {video_path}")
            return {"results": [], "video_info": {}}

        # ------ Read video metadata ------
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0

        video_info = {
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "duration": round(duration, 2),
        }
        print(f"🎬 VideoProcessor: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s, {width}x{height}")

        # ------ Select which frames to analyze ------
        frame_indices = self._select_frames(total_frames, fps)
        print(f"   Will analyze {len(frame_indices)} frames (interval strategy)")

        results = []
        processed = 0
        start_time = time.time()
        max_wall_time = 60.0  # never block longer than 60 seconds

        # Optional video writer for annotated output
        writer = None
        if output_path and generate_overlay:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for idx, frame_no in enumerate(frame_indices):
            # Wall-clock safety guard
            if time.time() - start_time > max_wall_time:
                print(f"   ⏱  Time limit reached at frame {frame_no} ({processed} processed)")
                break

            # Seek to the target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # Detect face and run model
            frame_result = self._process_single_frame(frame, frame_no, generate_overlay)

            if frame_result is not None:
                results.append(frame_result)

            processed += 1

            # Report progress
            progress = idx / len(frame_indices)
            if progress_callback:
                progress_callback(progress)

            # Yield CPU every 5 frames so FastAPI can handle other requests
            if processed % 5 == 0:
                await asyncio.sleep(0.001)

        cap.release()
        if writer:
            writer.release()

        elapsed = time.time() - start_time
        print(f"   ✅ Done: {len(results)} frames with faces analysed in {elapsed:.1f}s")

        # If no face was found in any frame, insert a fallback row so the UI
        # doesn't silently show an empty chart.
        if not results and total_frames > 0:
            print("   ⚠️  No faces detected – returning sentinel result")
            results = [{"frame": 0, "confidence_fake": 0.0, "no_face": True}]

        return {"results": results, "video_info": video_info}

    async def process_webcam(self, callback: Callable):
        """
        Real-time webcam processing loop.
        Receives frames from callback, runs face detection + model inference,
        and sends results back via callback.

        This method is designed to work with the WebSocket endpoint:
        the caller sends frames via WebSocket messages; this method
        generates continuous predictions.
        """
        frame_count = 0
        last_process_time = time.time()
        process_interval = 0.15  # analyse at ~6 fps to stay real-time

        try:
            while True:
                current_time = time.time()
                elapsed = current_time - last_process_time

                if elapsed < process_interval:
                    await asyncio.sleep(process_interval - elapsed)
                    continue

                # Create a dummy frame (the real frame comes via WebSocket data,
                # but here we allow the loop to produce a heartbeat result even
                # without a real frame until the client sends one).
                # The actual webcam frame path uses the shared _last_webcam_frame
                # attribute set by the WebSocket receiver.
                frame = getattr(self, "_last_webcam_frame", None)

                if frame is not None:
                    result = self._process_single_frame(frame, frame_count, generate_overlay=True)
                    if result is None:
                        result = {
                            "frame": frame_count,
                            "confidence_fake": 0.0,
                            "no_face": True,
                        }
                else:
                    # No frame yet – send a neutral heartbeat
                    result = {"frame": frame_count, "confidence_fake": 0.0, "waiting": True}

                await callback(result)
                last_process_time = current_time
                frame_count += 1
                await asyncio.sleep(0.05)

        except asyncio.CancelledError:
            print("Webcam processing cancelled")
        except Exception as e:
            print(f"Webcam processing error: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_frames(self, total_frames: int, fps: float) -> List[int]:
        """
        Choose frame indices to analyze.

        Strategy:
          - For short videos  (< 30 s)  → every 1 second
          - For medium videos (< 5 min) → every 2 seconds
          - For long videos             → every 5 seconds
          - Always include first + last frame
          - Capped at self.max_frames
        """
        if total_frames <= 0:
            return [0]

        if fps <= 0:
            fps = 30.0

        duration = total_frames / fps

        if duration < 30:
            step = max(1, int(fps * 0.2))    # every 0.2 s (~5 fps)
        elif duration < 120:
            step = max(1, int(fps * 0.5))    # every 0.5 s (2 fps)
        else:
            step = max(1, int(fps * 1.0))    # every 1.0 s (1 fps)

        indices = list(range(0, total_frames, step))

        # Guarantee first and last
        if 0 not in indices:
            indices.insert(0, 0)
        if total_frames - 1 not in indices:
            indices.append(total_frames - 1)

        # Cap total number
        if len(indices) > self.max_frames:
            # Sample evenly from the list
            idx_np = np.linspace(0, len(indices) - 1, self.max_frames, dtype=int)
            indices = [indices[i] for i in idx_np]

        return sorted(set(indices))

    def _process_single_frame(
        self,
        frame: np.ndarray,
        frame_no: int,
        generate_overlay: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Run face detection + model inference on one frame.
        Supports multiple faces per frame.
        """
        if self.face_detector is None or self.model is None:
            return None

        # Detect faces (frame is BGR from OpenCV)
        faces = self.face_detector.detect_faces(frame)
        if not faces:
            return None

        face_results = []
        max_fake_prob = 0.0

        for bbox in faces:
            face_crop = self.face_detector.extract_face(frame, bbox, target_size=self.model.INPUT_SIZE)
            if face_crop is None or face_crop.size == 0:
                continue

            try:
                conf = float(self.model.predict(face_crop))
                face_results.append({
                    "bbox": bbox,
                    "confidence": conf
                })
                if conf > max_fake_prob:
                    max_fake_prob = conf
            except Exception as e:
                print(f"   Model inference error on frame {frame_no}: {e}")

        if not face_results:
            return None

        result: Dict[str, Any] = {
            "frame": int(frame_no),
            "confidence_fake": round(float(max_fake_prob), 4),
            "face_count": len(faces),
        }

        # Optionally add annotated frame as base64 JPEG
        if generate_overlay:
            overlay = self._draw_overlay(frame, face_results)
            result["overlay_frame"] = self._cv2_to_base64(overlay)

        return result

    def _draw_overlay(
        self,
        frame: np.ndarray,
        face_results: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Draw bounding boxes and confidence labels for each face."""
        overlay = frame.copy()
        
        # Determine overall max confidence for the top-level status label
        overall_max = max([fr["confidence"] for fr in face_results]) if face_results else 0.0
        status_color = (0, 0, 255) if overall_max >= 0.5 else (0, 200, 0)
        status_text = f"{'FAKE DETECTED' if overall_max >= 0.5 else 'NORMAL'} ({overall_max:.1%})"

        for res in face_results:
            x, y, w, h = res["bbox"]
            conf = res["confidence"]
            color = (0, 0, 255) if conf >= 0.5 else (0, 200, 0)
            
            # Draw box around each face
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # Label each face specifically
            label = f"{conf:.0%}"
            cv2.putText(overlay, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Draw overall status banner in corner
        cv2.rectangle(overlay, (0, 0), (250, 40), (0, 0, 0), -1)
        cv2.putText(overlay, status_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
        return overlay

    @staticmethod
    def _cv2_to_base64(image: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf).decode("utf-8")

    @staticmethod
    def _base64_to_cv2(b64: str) -> np.ndarray:
        data = base64.b64decode(b64)
        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # ------------------------------------------------------------------
    # Legacy extract_frames (kept for backward compatibility)
    # ------------------------------------------------------------------

    async def extract_frames(self, callback=None):
        """Legacy method: extract frames from self.video_path."""
        if not self.video_path:
            return [], {}

        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        self.video_info = {
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "duration": duration,
        }

        indices = self._select_frames(total_frames, fps)
        frames = []

        for i, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, image = cap.read()
            if ok and image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frames.append({"frame": frame_idx, "image": image_rgb})
                if callback:
                    callback((i + 1) / len(indices))

        cap.release()
        return frames, self.video_info