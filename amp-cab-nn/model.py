import torch
import torch.nn as nn

class GuneAmp(nn.Module):
    def __init__(self):
        super().__init__()
        # Aggressive distortion model with hard clipping
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1, dilation=1),
            nn.Tanh(),
            
            nn.Conv1d(64, 64, 3, padding=2, dilation=2),
            nn.Tanh(),
            
            nn.Conv1d(64, 64, 3, padding=4, dilation=4),
            nn.Tanh(),
            
            nn.Conv1d(64, 64, 3, padding=8, dilation=8),
            nn.Tanh(),
            
            nn.Conv1d(64, 1, 3, padding=1)
        )
        
        # Gain boost to drive saturation harder
        self.gain = nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        y = self.net(x)
        
        # Apply gain to push into saturation
        y = y * self.gain
        
        # Hard clipping for aggressive distortion
        y = torch.clamp(y, -0.95, 0.95)
        
        return y[:, :, -1]