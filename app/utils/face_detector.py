import cv2
import numpy as np
from typing import List, Tuple, Optional
import os

class FaceDetector:
    """
    Real face detector using OpenCV's built-in face detection.
    Uses Haar Cascade as primary detector (no external model files needed).
    Optionally upgrades to DNN-based SSD detector if model files are present.
    """

    def __init__(self, min_detection_confidence: float = 0.5):
        self.min_detection_confidence = min_detection_confidence
        self._mode = "haar"  # 'haar' or 'dnn'
        self._dnn_net = None

        # Try to load DNN-based SSD face detector (more accurate)
        dnn_proto = os.path.join(os.path.dirname(__file__), "..", "models", "weights", "deploy.prototxt")
        dnn_model = os.path.join(os.path.dirname(__file__), "..", "models", "weights", "res10_300x300_ssd_iter_140000.caffemodel")
        if os.path.exists(dnn_proto) and os.path.exists(dnn_model):
            try:
                self._dnn_net = cv2.dnn.readNetFromCaffe(dnn_proto, dnn_model)
                self._mode = "dnn"
                print("✅ FaceDetector: Using DNN SSD face detector")
            except Exception as e:
                print(f"⚠️  FaceDetector: DNN load failed ({e}), falling back to Haar")

        if self._mode == "haar":
            # OpenCV built-in Haar cascade (always available)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(cascade_path)
            print("✅ FaceDetector: Using Haar Cascade face detector")

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.

        Args:
            image: BGR or RGB numpy array (H, W, 3)

        Returns:
            List of bounding boxes [(x, y, w, h), ...]
        """
        if image is None or image.size == 0:
            return []

        if self._mode == "dnn" and self._dnn_net is not None:
            return self._detect_dnn(image)
        else:
            return self._detect_haar(image)

    def _detect_dnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """DNN-based SSD face detection (more accurate, handles angles/lighting)."""
        h, w = image.shape[:2]
        # Ensure BGR for DNN
        if image.shape[2] == 3:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0)
            )
        else:
            return []

        self._dnn_net.setInput(blob)
        detections = self._dnn_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence >= self.min_detection_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    faces.append((x1, y1, x2 - x1, y2 - y1))

        return faces

    def _detect_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Haar Cascade face detection (fast, always available)."""
        if self._haar.empty():
            return []

        # Convert to grayscale for Haar Cascade
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.dtype == np.uint8 else cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Histogram equalization for better detection in varied lighting
        gray = cv2.equalizeHist(gray)

        faces_raw = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces_raw) == 0:
            return []

        # Return as list of tuples
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces_raw]

    def extract_face(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = (224, 224),
        margin: float = 0.3
    ) -> np.ndarray:
        """
        Extract and resize a face from an image, maintaining aspect ratio.
        Squares the crop before resizing to prevent distortion.
        """
        if image is None or image.size == 0:
            return np.array([])

        h_img, w_img = image.shape[:2]
        x, y, w, h = bbox

        # Calculate square box with margin
        cx, cy = x + w // 2, y + h // 2
        side = int(max(w, h) * (1 + margin * 2))
        
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w_img, x1 + side)
        y2 = min(h_img, y1 + side)
        
        # Adjust if hitting right/bottom edges to keep it as square as possible
        if x2 - x1 < side and x1 > 0: x1 = max(0, x2 - side)
        if y2 - y1 < side and y1 > 0: y1 = max(0, y2 - side)

        face = image[y1:y2, x1:x2]
        if face.size == 0:
            return np.array([])

        # Resize to model's expected input size
        face_resized = cv2.resize(face, target_size, interpolation=cv2.INTER_CUBIC)
        return face_resized

    def detect_and_extract_largest(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (224, 224)
    ) -> Optional[np.ndarray]:
        """
        Detect faces and return the largest one (most prominent face).
        Returns None if no face is found.
        """
        faces = self.detect_faces(image)
        if not faces:
            return None

        # Pick the largest face by area
        largest = max(faces, key=lambda b: b[2] * b[3])
        face = self.extract_face(image, largest, target_size)
        if face.size == 0:
            return None
        return face