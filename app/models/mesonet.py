import numpy as np
import os
from pathlib import Path

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False
    print("⚠️  onnxruntime not available – MesoNet will use fallback mode")


class MesoNet:
    """
    Real MesoNet implementation for deepfake detection.
    Uses the pre-trained ONNX model (mesonet.onnx) for actual inference.

    MesoNet architecture: 4-layer CNN specifically designed for face manipulation detection.
    Reference: Afchar et al. "MesoNet: a Compact Facial Video Forgery Detection Network" (2018)

    Input:  (1, 224, 224, 3)  float32, pixel values in [0, 1]
    Output: (1, 1)            float32, probability of being FAKE (0=real, 1=fake)
    """

    INPUT_SIZE = (224, 224)
    WEIGHTS_FILENAME = "mesonet.onnx"

    def __init__(self, weights_dir: str = None):
        self._session = None
        self._input_name = None
        self._output_name = None
        self._loaded = False

        # Resolve weights path
        if weights_dir is None:
            weights_dir = Path(__file__).resolve().parent / "weights"
        else:
            weights_dir = Path(weights_dir)

        weights_path = weights_dir / self.WEIGHTS_FILENAME

        if _ORT_AVAILABLE and weights_path.exists():
            self._load_onnx(str(weights_path))
        else:
            if not _ORT_AVAILABLE:
                print("⚠️  MesoNet: onnxruntime not installed")
            else:
                print(f"⚠️  MesoNet: weights not found at {weights_path}")

    def _load_onnx(self, weights_path: str):
        """Load ONNX model for inference."""
        try:
            # Prefer CPU provider for compatibility on Mac (MPS not supported by ORT yet)
            providers = ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(weights_path, providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            self._loaded = True
            print(f"✅ MesoNet: ONNX model loaded from {weights_path}")
            print(f"   Input:  {self._input_name} {self._session.get_inputs()[0].shape}")
            print(f"   Output: {self._output_name} {self._session.get_outputs()[0].shape}")
        except Exception as e:
            print(f"❌ MesoNet: Failed to load ONNX model: {e}")
            self._loaded = False

    def predict(self, face_image: np.ndarray) -> float:
        """
        Run deepfake detection on a single face image.

        Args:
            face_image: numpy array of shape (224, 224, 3), dtype uint8 or float32
                        Color format: BGR (from OpenCV) or RGB – model is robust to both.

        Returns:
            float: Probability that the face is FAKE (0.0 = definitely real, 1.0 = definitely fake)
        """
        if not self._loaded or self._session is None:
            raise RuntimeError("MesoNet model is not loaded")

        # Preprocess: ensure correct shape and dtype
        processed = self.prepare_image(face_image)

        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        batch = np.expand_dims(processed, axis=0)  # (1, 224, 224, 3)

        # Run inference
        outputs = self._session.run(
            [self._output_name],
            {self._input_name: batch}
        )

        # Output is (1, 1) -> extract scalar probability
        fake_prob = float(outputs[0][0][0])
        
        # Sensitivity boost (MesoNet is often too conservative)
        if fake_prob > 0.05:
            fake_prob = np.power(fake_prob, 0.65) # Stronger boost than Xception
            
        return float(np.clip(fake_prob, 0.0, 1.0))

    def prepare_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for MesoNet inference.

        Args:
            image: numpy array (H, W, 3), uint8 or float32

        Returns:
            numpy array (224, 224, 3), float32, values in [0, 1]
        """
        import cv2

        if image is None or image.size == 0:
            return np.zeros((224, 224, 3), dtype=np.float32)

        # Resize to model input size if needed
        if image.shape[:2] != (self.INPUT_SIZE[1], self.INPUT_SIZE[0]):
            image = cv2.resize(image, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)

        # Convert BGR (OpenCV) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)

        return image

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # Legacy compat: some callers use model.model.predict(x)
    @property
    def model(self):
        return self

    def __call__(self, face_image: np.ndarray) -> float:
        return self.predict(face_image)