import torch
import cv2
import numpy as np
from app.models.xceptionnet import XceptionNet

model = XceptionNet("app/models/weights")
# Test random image
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
prob = model.predict(img)
print("Rand img prob:", prob)

# test what happens if logit is negative or positive
t = model._preprocess(img)
print("Logit:", model._net(t).item())
