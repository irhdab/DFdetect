from app.models.mesonet import MesoNet
from app.models.xceptionnet import XceptionNet
from pathlib import Path

class ModelFactory:
    """
    Factory class for creating and managing deepfake detection models
    """
    
    # Available model types
    AVAILABLE_MODELS = ["mesonet", "xceptionnet"]
    
    def __init__(self, weights_dir=None):
        """
        Initialize the model factory
        
        Args:
            weights_dir: Directory to store model weights
        """
        self.weights_dir = Path(weights_dir) if weights_dir else None
    
    def create_model(self, model_name):
        """
        Create a model instance based on the model name
        
        Args:
            model_name: Name of the model to create
            
        Returns:
            Model instance
        """
        model_name = model_name.lower()
        
        if model_name == "mesonet":
            return MesoNet()
        elif model_name == "xceptionnet":
            return XceptionNet()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def get_model_info(self, model_name):
        """
        Get information about a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
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
    
    def get_available_models(self):
        """
        Get list of available models
        
        Returns:
            List of model information dictionaries
        """
        return [self.get_model_info(model) for model in self.AVAILABLE_MODELS] 