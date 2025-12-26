import torch
from model import GuneAmp
from audio_utils import pre_emphasis
from torch.optim import Adam
from torch.nn import MSELoss

def train_model(X, Y, device, epochs=50, batch_size=256, learn_rate=1e-4, window=256):
    model = GuneAmp().to(device)
    optimizer = Adam(model.parameters(), lr=learn_rate)
    mse = MSELoss()
    print("Training HAS BEGUN....")
    for epoch in range(epoch):
        perm = torch.randperm(len(X))
        total_loss = 0.0

        for i in range(0, len(X), batch_size):
            idx = perm[i:i+batch_size]
            xb = X[idx].to(device)
            yb = Y[idx].to(device)

            optimizer.zero_grad()
            yp = model(xb)

            loss = mse(yp, yb)
            loss += 0.5 * mse(
                pre_emphasis(yp.squeeze()),
                pre_emphasis(yb.squeeze())
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")
    return model