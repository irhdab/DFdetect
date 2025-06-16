import cv2
import numpy as np

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        """
        Initialize Face Detection (mock version for testing)
        
        Args:
            min_detection_confidence: Minimum confidence value for face detection
        """
        self.min_detection_confidence = min_detection_confidence
    
    def detect_faces(self, image):
        """
        Detect faces in an image (mock implementation)
        
        Args:
            image: RGB image numpy array
            
        Returns:
            List of face bounding boxes in format (x, y, w, h)
        """
        # For testing, just return a face in the center of the image
        height, width = image.shape[:2]
        face_width = width // 3
        face_height = height // 3
        x = (width - face_width) // 2
        y = (height - face_height) // 2
        
        # Return a mock face box
        return [(x, y, face_width, face_height)]
    
    def extract_face(self, image, bbox, target_size=(224, 224)):
        """
        Extract and resize a face from an image
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, w, h)
            target_size: Target size for face image
            
        Returns:
            Resized face image
        """
        x, y, w, h = bbox
        face = image[y:y+h, x:x+w]
        
        # Resize to target size
        face = cv2.resize(face, target_size)
        
        return face 