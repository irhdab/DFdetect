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
    
    def save_to_onnx(self, path):
        """Convert model to ONNX format for faster inference"""
        import tf2onnx
        import onnx
        
        # Create input signature
        signatures = [tf.TensorSpec((None, 224, 224, 3), tf.float32, name='input')]
        
        # Convert the model
        onnx_model, _ = tf2onnx.convert.from_keras(self.model, input_signature=signatures, opset=13)
        
        # Save the model
        onnx.save(onnx_model, path)
        
        return path 