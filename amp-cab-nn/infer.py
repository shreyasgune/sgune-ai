import torch
from tqdm import tqdm
from audio_utils import apply_ir, downsample
import torchaudio

def run_inference(
    model,
    clean_os,
    cab_ir,
    oversample_factor,
    window,
    device,
    sr,
    output_file="modeled.wav",
    batch_size=32
):
    model = model.to(device)
    model.eval()

    clean_os = clean_os.to(device)

    N = clean_os.shape[0]
    out_os = torch.zeros(N, device=device)
    weight = torch.zeros(N, device=device)

    with torch.no_grad():
        for i in tqdm(range(0, N - window, batch_size), desc="running inference"):
            # Batch multiple inference windows for GPU efficiency
            batch_end = min(i + batch_size, N - window)
            batch_size_actual = batch_end - i
            
            # Extract batch of windows
            windows = []
            for j in range(i, batch_end):
                x = clean_os[j:j+window].unsqueeze(0)  # [1, window]
                windows.append(x)
            
            x_batch = torch.cat(windows, dim=0).unsqueeze(1)  # [batch_size, 1, window]
            y_batch = model(x_batch)  # [batch_size, 1, 1]
            
            # Apply results back with overlap-add
            for j, idx in enumerate(range(i, batch_end)):
                y = y_batch[j].squeeze()
                out_os[idx:idx+window] += y
                weight[idx:idx+window] += 1.0

    # normalize overlap-add
    out_os = out_os / weight.clamp(min=1.0)

    # downsample only if oversampling was actually used
    if oversample_factor > 1:
        out = downsample(out_os, oversample_factor)
        out_sr = sr
    else:
        out = out_os
        out_sr = sr

    # apply cabinet IR
    out = apply_ir(out, cab_ir.to(device))
    print("After IR length:", out.shape[0])


    # normalize safely
    out = out / out.abs().max().clamp(min=1e-8)

    out_cpu = out.detach().cpu()

    print("OUT stats:",
          out.min().item(),
          out.max().item(),
          out.abs().mean().item())

    print("Lengths:",
          len(clean_os),
          len(out_os),
          len(out_cpu))

    torchaudio.save(output_file, out_cpu.unsqueeze(0), out_sr)
    print(f"Saved output: {output_file}")

    return out_cpu
