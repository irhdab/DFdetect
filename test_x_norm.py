import torch
import cv2
import numpy as np
from app.models.xceptionnet import XceptionNet

model = XceptionNet("app/models/weights")
img = np.random.randint(0, 255, (299, 299, 3), dtype=np.uint8)

# Test Inception normalization manually
model._net.eval()
with torch.no_grad():
    # Current (ImageNet)
    t_imgnet = model._preprocess(img)
    p_imgnet = torch.sigmoid(model._net(t_imgnet)).item()
    
    # Inception (-1 to 1)
    img_inc = (img.astype(np.float32) / 127.5) - 1.0
    t_inc = torch.from_numpy(img_inc.transpose(2,0,1)).unsqueeze(0)
    p_inc = torch.sigmoid(model._net(t_inc)).item()
    
    # Simple 0 to 1
    t_simple = torch.from_numpy((img.astype(np.float32)/255.0).transpose(2,0,1)).unsqueeze(0)
    p_simple = torch.sigmoid(model._net(t_simple)).item()

print(f"ImageNet: {p_imgnet}")
print(f"Inception: {p_inc}")
print(f"Simple: {p_simple}")
