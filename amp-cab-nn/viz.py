import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_waveform(x, sr, title, seconds=0.05):
    n = int(sr * seconds)
    t = np.arange(n) / sr
    plt.figure(figsize=(8, 3))
    plt.plot(t, x[:n])
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitube")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_spectrum(x, sr, title):
    X = torch.fft.rfft(x)
    mag = 20 * torch.log10(torch.abs(X) + 1e-8)
    freqs = torch.fft.rfftfreq(len(x), 1 / sr)

    plt.figure(figsize=(8,3))
    plt.plot(freqs.numpy(), mag.numpy())
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xlim(20, sr/2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_ir(ir, sr, title):
    t = np.arange(len(ir)) / sr
    plt.figure(figsize=(8,3))
    plt.plot(t, ir)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_transfer_curve(model, window, device):
    x = torch.linspace(-1, 1, 2000)
    y = []

    model.eval()
    with torch.no_grad():
        for i in range(window, len(x)):
            window_tensor = x[i-window:i].unsqueeze(0).unsqueeze(0).to(device)
            y.append(model(window_tensor).cpu())

    y = torch.stack(y).squeeze()

    plt.figure(figsize=(8,3))
    plt.plot(x[window:], y)
    plt.title("Learned Transfer Curve")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    