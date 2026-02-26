import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchcv.model_provider import get_model

class CustomXception(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.ModuleList([get_model('xception').features])
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.b1 = nn.BatchNorm1d(2048)
                self.l = nn.Linear(2048, 512)
                self.b2 = nn.BatchNorm1d(512)
                self.o = nn.Linear(512, 1)
                
            def forward(self, x):
                x = self.b1(x)
                x = F.relu(self.l(x))  # or something
                x = self.b2(x)
                x = self.o(x)
                return x
        self.h1 = Head()
        
    def forward(self, x):
        x = self.base[0](x)
        x = self.pool(x).flatten(1)
        x = self.h1(x)
        return x

m = CustomXception()
ckpt = torch.load('app/models/weights/xceptionnet.pth', map_location='cpu')
m.load_state_dict(ckpt, strict=False)
print("Load success!")
