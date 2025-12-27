import torch
from tqdm import tqdm
from model import GuneAmp
from audio_utils import pre_emphasis
from torch.optim import Adam
from torch.nn import MSELoss
from torch.amp import autocast, GradScaler
import os

# enable cuDNN autotuner for potentially faster convs on CUDA
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

def train_model(X, Y, device, epochs, batch_size=256, learn_rate=5e-5, window=256, checkpoint_path="checkpoints/gune_amp.pt", resume=True):
    model = GuneAmp().to(device)
    optimizer = Adam(model.parameters(), lr=learn_rate)
    mse = MSELoss()
    scaler = GradScaler()  # For mixed precision training

    start_epoch = 0
    if resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")

    print("Training HAS BEGUN....")
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        total_loss = 0.0

        for i in tqdm(range(0, len(X), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            idx = perm[i:i+batch_size]
            xb = X[idx].to(device)
            yb = Y[idx].to(device)

            optimizer.zero_grad()
            
            # Mixed precision training for faster computation and lower memory usage
            with autocast('cuda'):
                yp = model(xb)
                loss = mse(yp, yb)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

        #Checkpointing
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": total_loss
        }, checkpoint_path)

    return model