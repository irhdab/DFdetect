from app.models.mesonet import MesoNet
from app.models.xceptionnet import XceptionNet
from pathlib import Path


class ModelFactory:
    """
    Factory class for creating and managing real deepfake detection models.
    """

    AVAILABLE_MODELS = ["mesonet", "xceptionnet"]

    def __init__(self, weights_dir=None):
        self.weights_dir = Path(weights_dir) if weights_dir else Path(__file__).parent / "weights"
        self._cache: dict = {}

    def create_model(self, model_name: str):
        """
        Return a loaded model instance (cached after first load).
        """
        model_name = model_name.lower()

        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {self.AVAILABLE_MODELS}")

        if model_name in self._cache:
            return self._cache[model_name]

        if model_name == "mesonet":
            instance = MesoNet(weights_dir=str(self.weights_dir))
        elif model_name == "xceptionnet":
            instance = XceptionNet(weights_dir=str(self.weights_dir))
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self._cache[model_name] = instance
        return instance

    def get_model_info(self, model_name: str) -> dict:
        model_name = model_name.lower()
        info_map = {
            "mesonet": {
                "id": "mesonet",
                "name": "MesoNet",
                "description": "Lightweight 4-layer CNN optimized for face manipulation detection (fast)",
                "loaded": False,
            },
            "xceptionnet": {
                "id": "xceptionnet",
                "name": "XceptionNet",
                "description": "EfficientNet-B4 backbone for deepfake detection (more accurate)",
                "loaded": False,
            },
        }

        info = info_map.get(model_name, {
            "id": "unknown",
            "name": "Unknown Model",
            "description": "Unknown model type",
            "loaded": False,
        })

        # Report real loaded status if model is cached
        if model_name in self._cache:
            info = info.copy()
            info["loaded"] = getattr(self._cache[model_name], "is_loaded", False)

        return info

    def get_available_models(self) -> list:
        return [self.get_model_info(m) for m in self.AVAILABLE_MODELS]