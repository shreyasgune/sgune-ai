import os
import sys
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# -------------------------
# CONFIG
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 44100

WINDOW = 256                #  larger context
PRED_SAMPLES = 32            #  multi-sample prediction
BATCH_SIZE = 256
EPOCHS = 1
LEARNING_RATE = 1e-4

CHECKPOINT_PATH = "checkpoints/metal_amp_v2.pt"

torch.backends.cudnn.benchmark = True
os.makedirs("checkpoints", exist_ok=True)

# -------------------------
# MODEL (Wide + Aggressive)
# -------------------------
class MetalAmpNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2),
            nn.LeakyReLU(0.2),

            nn.Conv1d(64, 128, 5, padding=2),
            nn.LeakyReLU(0.2),

            nn.Conv1d(128, 256, 5, padding=2),
            nn.LeakyReLU(0.2),

            nn.Conv1d(256, 256, 5, padding=2),
            nn.LeakyReLU(0.2),

            nn.Conv1d(256, 1, 1)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# DATA PREP
# -------------------------
# def make_windows(clean, amp, window):
#     X, Y = [], []
#     for i in range(len(clean) - window - PRED_SAMPLES):
#         X.append(clean[i:i+window])
#         Y.append(amp[i+window:i+window+PRED_SAMPLES])
#     return torch.stack(X), torch.stack(Y)

class RandomWindowDataset(torch.utils.data.Dataset):
    def __init__(self, clean, amp, window, pred):
        self.clean = clean
        self.amp = amp
        self.window = window
        self.pred = pred
        self.max_i = len(clean) - window - pred

    def __len__(self):
        return self.max_i

    def __getitem__(self, idx):
        i = torch.randint(0, self.max_i, (1,)).item()
        x = self.clean[i:i+self.window]
        y = self.amp[i+self.window:i+self.window+self.pred]
        return x.unsqueeze(0), y



# -------------------------
# JCM-STYLE ASYMMETRIC LOSS
# -------------------------
def asymmetric_clipping_loss(pred, target):
    """
    Penalizes negative half-cycle mismatch harder than positive,
    mimicking Marshall-style asymmetric tube clipping.
    """
    error = pred - target
    pos = torch.relu(error)
    neg = torch.relu(-error)

    return (pos.mean() + 2.0 * neg.mean())

# -------------------------
# TRAIN
# -------------------------
def train(clean_wav, amp_wav):
    print("Training mode")

    model = MetalAmpNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mse = nn.MSELoss()

    if os.path.exists(CHECKPOINT_PATH):
        print("Resuming from checkpoint")
        ckpt = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])

    clean, _ = torchaudio.load(clean_wav)
    amp, _ = torchaudio.load(amp_wav)

    clean = clean[0]
    amp = amp[0]

    # X, Y = make_windows(clean, amp, WINDOW)
    # X = X.unsqueeze(1)
    # Y = Y

    ds = RandomWindowDataset(clean, amp, WINDOW, PRED_SAMPLES)
    # ds = TensorDataset(X, Y)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    scaler = torch.amp.GradScaler(device="cuda")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for xb, yb in tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad(set_to_none=True)
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)


            with torch.amp.autocast(device_type="cuda"):
                pred = model(xb)                       # [B,1,WINDOW]
                pred_chunk = pred[:, 0, -PRED_SAMPLES:]  # [B,32]

                time_loss = mse(pred_chunk, yb)
                clip_loss = asymmetric_clipping_loss(pred_chunk, yb)

                loss = time_loss + 0.5 * clip_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Loss: {total_loss / len(dl):.6f}")

        torch.save({
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "window": WINDOW,
            "pred_samples": PRED_SAMPLES
        }, CHECKPOINT_PATH)

    print("Training complete")

# -------------------------
# INFERENCE
# -------------------------
def infer(clean_wav, out_wav):
    print("Inference mode")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = MetalAmpNet().to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    clean, _ = torchaudio.load(clean_wav)
    clean = clean[0].to(DEVICE)

    out = torch.zeros_like(clean)

    with torch.no_grad():
        for i in tqdm(range(WINDOW, len(clean) - PRED_SAMPLES)):
            x = clean[i-WINDOW:i].unsqueeze(0).unsqueeze(0)
            y = model(x)
            out[i:i+PRED_SAMPLES] = y[0, 0, -PRED_SAMPLES:]

    #  post-drive shaping
    out = torch.tanh(out * 3.5)
    out = out / out.abs().max()

    torchaudio.save(out_wav, out.unsqueeze(0).cpu(), SR)
    print(f"Saved: {out_wav}")

# -------------------------
# ENTRY
# -------------------------
if __name__ == "__main__":
    mode = sys.argv[1]

    if mode == "train":
        train("clean.wav", "amp.wav")
    elif mode == "infer":
        infer("clean.wav", "metal_output.wav")
    else:
        print("Usage: python metal_amp.py [train|infer]")
