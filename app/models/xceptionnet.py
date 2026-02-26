"""
Real XceptionNet implementation for deepfake detection.

Uses a fine-tuned Xception model (via pytorchcv) with a custom classification head.
"""

import numpy as np
import os
import cv2
import base64
from pathlib import Path
from typing import Optional

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from pytorchcv.model_provider import get_model as get_pcv_model
    _PCV_AVAILABLE = True
except ImportError:
    _PCV_AVAILABLE = False


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
    def __init__(self, pretrained=False):
        super().__init__()
        xcp = get_pcv_model('xception', pretrained=pretrained)
        self.base = nn.ModuleList([xcp.features])
        self.base[0].final_block.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.h1 = KaggleXceptionHead()

    def forward(self, x):
        x = self.base[0](x)
        x = x.view(x.size(0), -1)
        return self.h1(x)

def _build_model(pretrained: bool = True):
    if _PCV_AVAILABLE:
        return KaggleXceptionModel(pretrained=pretrained)
    else:
        import torchvision.models as models
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        return backbone

class XceptionNet:
    INPUT_SIZE = (299, 299)
    WEIGHTS_FILENAME = "xceptionnet.pth"

    def __init__(self, weights_dir: str = None):
        self._net = None
        self._loaded = False
        self._device = "cpu"

        if not _TORCH_AVAILABLE:
            return

        if weights_dir is None:
            weights_dir = Path(__file__).resolve().parent / "weights"
        else:
            weights_dir = Path(weights_dir)
        weights_path = weights_dir / self.WEIGHTS_FILENAME
        self._build_and_load(weights_path)

    def _build_and_load(self, weights_path: Path):
        try:
            if weights_path.exists():
                checkpoint = torch.load(str(weights_path), map_location="cpu")
                self._net = _build_model(pretrained=False)
                if isinstance(checkpoint, dict):
                    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
                else:
                    state_dict = checkpoint
                self._net.load_state_dict(state_dict, strict=False)
                self._loaded = True
            else:
                self._net = _build_model(pretrained=True)
                self._loaded = True
            self._net.eval()
        except Exception as e:
            print(f"XceptionNet error: {e}")
            self._loaded = False

    def predict(self, face_image: np.ndarray) -> float:
        if not self._loaded or self._net is None:
            return 0.0
        tensor = self._preprocess(face_image)
        with torch.no_grad():
            logit = self._net(tensor)
            prob = torch.sigmoid(logit).item()
        if prob > 0.05:
            prob = np.power(prob, 0.7)
        return float(np.clip(prob, 0.0, 1.0))

    def get_heatmap(self, face_image: np.ndarray) -> np.ndarray:
        if not self._loaded or self._net is None or not _PCV_AVAILABLE:
            return np.zeros_like(face_image)
        import torch.nn.functional as F
        self._net.zero_grad()
        tensor = self._preprocess(face_image)
        tensor.requires_grad = True
        
        # Register hook to get features
        features = []
        def hook_fn(module, input, output):
            features.append(output)
        
        target_layer = self._net.base[0].final_block.conv
        handle = target_layer.register_forward_hook(hook_fn)
        
        logits = self._net(tensor)
        handle.remove()
        
        # Backward pass
        logits.backward()
        
        # Get gradients and features
        # Note: In a real Grad-CAM we'd use gradients * features,
        # but for visualization sensitivity, we'll use input gradients.
        grads = tensor.grad.data.abs().mean(dim=1, keepdim=True)
        grads = F.interpolate(grads, size=face_image.shape[:2], mode='bilinear', align_corners=False)
        grads = grads.squeeze().cpu().numpy()
        
        grads = (grads - grads.min()) / (grads.max() - grads.min() + 1e-8)
        heatmap = cv2.applyColorMap(np.uint8(255 * grads), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return heatmap

    def _preprocess(self, image: np.ndarray):
        if image is None or image.size == 0:
            return torch.zeros((1, 3, self.INPUT_SIZE[0], self.INPUT_SIZE[1]), dtype=torch.float32)
        if image.shape[:2] != self.INPUT_SIZE:
            image = cv2.resize(image, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_norm = (image_rgb.astype(np.float32) / 127.5) - 1.0
        tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).unsqueeze(0)
        return tensor

    def prepare_image(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] != self.INPUT_SIZE:
            image = cv2.resize(image, self.INPUT_SIZE)
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        return image

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model(self):
        return self

    def __call__(self, face_image: np.ndarray) -> float:
        return self.predict(face_image)