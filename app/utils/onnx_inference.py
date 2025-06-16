import numpy as np

class ONNXInference:
    """
    Mock ONNX inference implementation for testing
    """
    
    def __init__(self, model_path, input_name=None, output_name=None):
        """
        Initialize ONNX inference
        
        Args:
            model_path: Path to ONNX model
            input_name: Name of input node
            output_name: Name of output node
        """
        self.model_path = model_path
        self.input_name = input_name or "input"
        self.output_name = output_name or "output"
    
    def prepare_image(self, image):
        """
        Prepare image for inference
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Preprocessed image ready for inference
        """
        # In mock mode, just return the image
        return image
    
    def predict(self, input_data):
        """
        Run inference on input data
        
        Args:
            input_data: Input data as numpy array
            
        Returns:
            Prediction result
        """
        # For testing, return random prediction
        return np.random.random((1, 1)) 