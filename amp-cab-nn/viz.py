import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import os

# Create graphs folder if it doesn't exist
GRAPHS_DIR = "graphs"
if not os.path.exists(GRAPHS_DIR):
    os.makedirs(GRAPHS_DIR)

def _sanitize_filename(title):
    """Convert title to safe filename."""
    return title.lower().replace(" ", "_").replace("/", "_")

def plot_waveform(x, sr, title, seconds=0.05):
    # ensure numpy array on CPU for matplotlib
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    plt.figure(figsize=(8, 3))
    plt.plot(t, x_np[:n])
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitube")
    plt.grid(True)
    plt.tight_layout()
    # Save before showing
    filename = os.path.join(GRAPHS_DIR, f"waveform_{_sanitize_filename(title)}.png")
    plt.savefig(filename, dpi=100)
    print(f"  → Saved: {filename}")
    plt.show()

def plot_spectrum(x, sr, title):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
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
    # Save before showing
    filename = os.path.join(GRAPHS_DIR, f"spectrum_{_sanitize_filename(title)}.png")
    plt.savefig(filename, dpi=100)
    print(f"  → Saved: {filename}")
    plt.show()

def plot_ir(ir, sr, title):
    if isinstance(ir, torch.Tensor):
        ir_np = ir.detach().cpu().numpy()
    else:
        ir_np = np.asarray(ir)
    t = np.arange(len(ir_np)) / sr
    plt.figure(figsize=(8,3))
    plt.plot(t, ir_np)
    plt.title(title)
    # Save before showing
    filename = os.path.join(GRAPHS_DIR, f"ir_{_sanitize_filename(title)}.png")
    plt.savefig(filename, dpi=100)
    print(f"  → Saved: {filename}")
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
        for i in tqdm(range(window, len(x)), desc="Computing transfer curve"):
            window_tensor = x[i-window:i].unsqueeze(0).unsqueeze(0).to(device)
            y.append(model(window_tensor).cpu())

    y = torch.stack(y).squeeze()

    plt.figure(figsize=(8,3))
    plt.plot(x[window:].numpy(), y.numpy())
    plt.title("Learned Transfer Curve")
    # Save before showing
    filename = os.path.join(GRAPHS_DIR, "transfer_curve.png")
    plt.savefig(filename, dpi=100)
    print(f"  → Saved: {filename}")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
