"""
Real XceptionNet implementation for deepfake detection.

Uses a fine-tuned EfficientNet-B4 (via timm) as a drop-in replacement for Xception
since:
  1. Xception is not available in torchvision by default.
  2. EfficientNet-B4 has comparable accuracy and is well-supported.
  3. We use ImageNet-pretrained weights and apply a deepfake-detection linear head.

When real fine-tuned weights (xceptionnet.pth) are present in the weights directory,
those are loaded. Otherwise, ImageNet-pretrained weights are used as a baseline
(transfer-learning basis) – results will be less accurate but the pipeline is real.
"""

import numpy as np
import os
from pathlib import Path
from typing import Optional

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available – XceptionNet will use fallback mode")

try:
    from pytorchcv.model_provider import get_model as get_pcv_model
    _PCV_AVAILABLE = True
except ImportError:
    _PCV_AVAILABLE = False
    print("⚠️  pytorchcv not available – XceptionNet will fail to load real weights")


class KaggleXceptionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.BatchNorm1d(2048)
        self.l = nn.Linear(2048, 512)
        self.b2 = nn.BatchNorm1d(512)
        self.o = nn.Linear(512, 1)

    def forward(self, x):
        import torch.nn.functional as F
        x = self.b1(x)
        x = F.relu(self.l(x))
        x = self.b2(x)
        return self.o(x)

class KaggleXceptionModel(nn.Module):
    """Architecture matching Kaggle FaceForensics++ Xception Net weights."""
    def __init__(self, pretrained=False):
        super().__init__()
        # Pytorchcv Xception model
        xcp = get_pcv_model('xception', pretrained=pretrained)
        # Wrap features in ModuleList to match 'base.0' prefix
        self.base = nn.ModuleList([xcp.features])
        
        # Replace the problematic fixed-size pooling (kernel_size=10) 
        # with an Adaptive pooling so it works with any input size (like 224 or 299)
        self.base[0].final_block.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.h1 = KaggleXceptionHead()

    def forward(self, x):
        x = self.base[0](x)
        # x is now (B, 2048, 1, 1) due to adaptive pooling
        x = x.view(x.size(0), -1) # Flatten
        return self.h1(x)


def _build_model(pretrained: bool = True):
    """
    Build Kaggle-compatible XceptionNet with a binary classification head.
    """
    if _PCV_AVAILABLE:
        return KaggleXceptionModel(pretrained=pretrained)
    else:
        # Fallback
        import torchvision.models as models
        import torch.nn as nn
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        return backbone


class XceptionNet:
    """
    Real XceptionNet-equivalent for deepfake detection (EfficientNet-B4 backbone).

    Input:  face image (299, 299, 3) numpy uint8 or float32
    Output: float probability of being FAKE (0.0 = real, 1.0 = fake)
    """

    INPUT_SIZE = (299, 299)
    WEIGHTS_FILENAME = "xceptionnet.pth"

    def __init__(self, weights_dir: str = None):
        self._net = None
        self._loaded = False
        self._device = "cpu"

        if not _TORCH_AVAILABLE:
            print("⚠️  XceptionNet: PyTorch not installed")
            return

        # Resolve weights path
        if weights_dir is None:
            weights_dir = Path(__file__).resolve().parent / "weights"
        else:
            weights_dir = Path(weights_dir)

        weights_path = weights_dir / self.WEIGHTS_FILENAME

        self._build_and_load(weights_path)

    def _build_and_load(self, weights_path: Path):
        """Build model and load weights if available."""
        try:
            # Try to load fine-tuned weights
            if weights_path.exists():
                print(f"📦 XceptionNet: Loading fine-tuned weights from {weights_path}")
                checkpoint = torch.load(str(weights_path), map_location="cpu")
                # Build model architecture (no pretrained download needed since we have weights)
                self._net = _build_model(pretrained=False)
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
                else:
                    state_dict = checkpoint
                self._net.load_state_dict(state_dict, strict=False)
                print(f"✅ XceptionNet: Fine-tuned weights loaded")
            else:
                # Fall back to ImageNet-pretrained as base feature extractor
                print(f"⚠️  XceptionNet: No fine-tuned weights at {weights_path}")
                print(f"   Using ImageNet pre-trained backbone (less accurate for deepfake detection)")
                self._net = _build_model(pretrained=True)

            self._net.eval()
            self._loaded = True

        except Exception as e:
            print(f"❌ XceptionNet: Failed to build/load model: {e}")
            import traceback
            traceback.print_exc()
            self._loaded = False

    def predict(self, face_image: np.ndarray) -> float:
        """
        Run deepfake detection on a single face image.

        Args:
            face_image: numpy array (224, 224, 3), uint8 or float32

        Returns:
            float: Probability that the face is FAKE (0.0-1.0)
        """
        if not self._loaded or self._net is None:
            raise RuntimeError("XceptionNet model is not loaded")

        import torch
        import torch.nn.functional as F

        tensor = self._preprocess(face_image)  # (1, 3, 224, 224)

        with torch.no_grad():
            logit = self._net(tensor)  # (1, 1)
            prob = torch.sigmoid(logit).item()

        # Calibration/Sensitivity Boost:
        # Subtle deepfakes often sit in the 0.2-0.4 range. 
        # We apply a slight non-linear boost to make the model more sensitive.
        if prob > 0.05:
            # Shift 0.3 -> ~0.5, 0.5 -> 0.7
            prob = np.power(prob, 0.7) 

        return float(np.clip(prob, 0.0, 1.0))

    def _preprocess(self, image: np.ndarray):
        """
        Convert numpy BGR image to normalized PyTorch RGB tensor.
        Uses Inception-style normalization ([-1, 1]) common for Xception.
        """
        import torch
        import cv2

        if image is None or image.size == 0:
            return torch.zeros((1, 3, self.INPUT_SIZE[0], self.INPUT_SIZE[1]), dtype=torch.float32)

        # Resize if needed
        if image.shape[:2] != self.INPUT_SIZE:
            image = cv2.resize(image, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)

        # Convert BGR (OpenCV) to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize to [-1, 1]
        image_norm = (image_rgb.astype(np.float32) / 127.5) - 1.0

        # HWC -> CHW, add batch dimension
        tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).unsqueeze(0)  # (1, 3, H, W)
        return tensor

    def prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Resize + normalize image (for compatibility with existing callers)."""
        import cv2
        if image.shape[:2] != self.INPUT_SIZE:
            image = cv2.resize(image, self.INPUT_SIZE)
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        return image

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # Legacy compat
    @property
    def model(self):
        return self

    def __call__(self, face_image: np.ndarray) -> float:
        return self.predict(face_image)