import torch
import torch.nn as nn

class GuneAmp(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1, dilation=1),
            nn.Tanh(),
            nn.Conv1d(32, 32, 3, padding=2, dilation=2),
            nn.Tanh(),
            nn.Conv1d(32, 32, 3, padding=4, dilation=4),
            nn.Tanh(),
            nn.Conv1d(32, 1, 3, padding=1)
        )

    def forward(self, x):
        y = self.net(x)
        return y[:, :, -1]