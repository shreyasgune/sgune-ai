import torch
import torch.nn as nn

class GuneAmp(nn.Module):
    def __init__(self):
        super().__init__()
        # Increased capacity: more filters and deeper network for aggressive distortion
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1, dilation=1),
            nn.Tanh(),
            
            nn.Conv1d(64, 64, 3, padding=2, dilation=2),
            nn.Tanh(),
            
            nn.Conv1d(64, 64, 3, padding=4, dilation=4),
            nn.Tanh(),
            
            # Additional layer for deeper feature extraction
            nn.Conv1d(64, 64, 3, padding=8, dilation=8),
            nn.Tanh(),
            
            nn.Conv1d(64, 1, 3, padding=1)
        )

    def forward(self, x):
        y = self.net(x)
        return y[:, :, -1]