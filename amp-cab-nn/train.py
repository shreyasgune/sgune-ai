import torch
from tqdm import tqdm
from model import GuneAmp
from audio_utils import pre_emphasis
from torch.optim import Adam
from torch.nn import MSELoss

# enable cuDNN autotuner for potentially faster convs on CUDA
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

def train_model(X, Y, device, epochs, batch_size=256, learn_rate=5e-5, window=256):
    model = GuneAmp().to(device)
    optimizer = Adam(model.parameters(), lr=learn_rate)
    mse = MSELoss()
    print("Training HAS BEGUN....")
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        total_loss = 0.0

        for i in tqdm(range(0, len(X), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            idx = perm[i:i+batch_size]
            xb = X[idx].to(device)
            yb = Y[idx].to(device)

            optimizer.zero_grad()
            yp = model(xb)

            # Reconstruction loss
            loss = mse(yp, yb)
            
            # Pre-emphasis loss (weight 2.0 for maximum distortion harmonics)
            loss += 2.0 * mse(
                pre_emphasis(yp.squeeze()),
                pre_emphasis(yb.squeeze())
            )
            
            # Harmonic distortion loss: encourage outputs to follow saturation curves
            # This penalizes outputs that are too linear and encourages non-linear behavior
            yp_sq = yp ** 2
            yb_sq = yb ** 2
            loss += 0.3 * mse(yp_sq, yb_sq)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")
    return model