import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from .ResNet50 import ResNet50
from .Transformer import TransformerEncoderOnly

class MainNetwork(nn.Module):
    def __init__(self, num_class):
        super(MainNetwork, self).__init__()
        self.extract_features = ResNet50(num_class) # (B, T, num_class))
        self.transformer = TransformerEncoderOnly(num_class, n_encoders=2) # (B, num_class)
        self.fc1 = nn.Linear(num_class, 1) # (B, 1)
        self.fc2 = nn.Linear(num_class, 1) # (B, 1)
        self.fc3 = nn.Linear(num_class, 2560) # (B, 2560)
        self.fc4 = nn.Linear(num_class, 2560) # (B, 2560)
                             
    @autocast()
    def forward(self, x):
        B, T, C, H, W = x.shape
        out = self.extract_features(x)
        out = self.transformer(out)
        ret = [
            # Arousal / Valence (B, 2)
            torch.cat([self.fc1(out),self.fc2(out)], dim=1),
            # ECG L/R (B, 2, 2560)
            torch.cat([torch.reshape(self.fc3(out), (B, 1, 2560)), torch.reshape(self.fc4(out), (B, 1, 2560))], dim=1)
        ]
        return ret