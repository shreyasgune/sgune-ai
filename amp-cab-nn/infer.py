import torch
from tqdm import tqdm
from audio_utils import apply_ir, downsample
import torchaudio

def run_inference(model, clean_os, cab_ir, oversample_factor, window, device, sr, output_file="modeled.wav"):
    model = model.to(device)
    model.eval()
    # ensure working tensors are on the device for speed
    out_os = clean_os.clone().to(device)

    with torch.no_grad():
        for i in tqdm(range(window, len(clean_os)), desc="running inference"):
            x = clean_os[i-window:i].unsqueeze(0).unsqueeze(0).to(device)
            val = model(x)
            # squeeze/broadcast to match 1D storage
            out_os[i] = val.squeeze().to(device)

    # downsample / apply IR on device
    out = downsample(out_os, oversample_factor)
    out = apply_ir(out, cab_ir.to(device))
    out = out / out.abs().max()

    # move to CPU for saving and visualization
    out_cpu = out.detach().cpu()
    torchaudio.save(output_file, out_cpu.unsqueeze(0), sr)
    print(f"Saved output: {output_file}")
    return out_cpu
