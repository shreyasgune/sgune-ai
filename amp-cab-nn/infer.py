import torch
from audio_utils import apply_ir, downsample
import torchaudio

def run_inference(model, clean_os, cab_ir, oversample_factor, window, device, sr, output_file="modeled.wav"):
    model.eval()
    out_os = clean_os.clone()

    with torch.no_grad():
        for i in range(window, len(clean_os)):
            x = clean_os[i-window:i].unsqueeze(0).unsqueeze(0).to(device)
            out_os[i] = model(x).cpu()

    out = downsample(out_os, oversample_factor)
    out = apply_ir(out, cab_ir)
    out = out / out.abs().max()

    torchaudio.save(output_file, out.unsqueeze(0), sr)
    print(f"Saved output: {output_file}")
    return out
