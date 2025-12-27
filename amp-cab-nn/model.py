import torch
import torch.nn as nn

class GuneAmp(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1),
            nn.Tanh(),
            nn.Conv1d(64, 64, 3, padding=2, dilation=2),
            nn.Tanh(),
            nn.Conv1d(64, 64, 3, padding=4, dilation=4),
            nn.Tanh(),
            nn.Conv1d(64, 64, 3, padding=8, dilation=8),
            nn.Tanh(),
            nn.Conv1d(64, 1, 3, padding=1)
        )

        self.gain = nn.Parameter(torch.tensor(3.0))

    def forward(self, x):
        y = self.net(x)

        # take LAST sample only
        y = y[:, :, -1:]          # (B,1,1)

        # residual connection (CRITICAL)
        y = y + x[:, :, -1:]

        y = torch.tanh(y * self.gain)
        return y