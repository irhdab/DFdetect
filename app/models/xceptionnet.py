import numpy as np

# Mock tensorflow version
class Model:
    def predict(self, x, verbose=0):
        # Generate low confidence values for real videos (0.05 to 0.25)
        # This indicates the model is more confident these are real, not deepfakes
        # XceptionNet is slightly more accurate than MesoNet
        return np.random.uniform(0.05, 0.25, (1, 1))

class XceptionNet:
    """
    Mock implementation of XceptionNet for deepfake detection
    """
    
    def __init__(self):
        self.model = Model()
    
    def build(self):
        """Build the XceptionNet architecture"""
        # Mock model
        self.model = Model()
        return self.model
    
    def prepare_image(self, image):
        """Prepare image for inference"""
        # Just return the image for mock testing
        return image
    async def save_to_onnx(self, output_path: str):
        """Mock method for saving to ONNX format"""
        # In a real implementation, we would use tf2onnx to convert the model
        print(f"Mock: Saving XceptionNet model to ONNX at {output_path}")
        await asyncio.sleep(0.5)
        return True