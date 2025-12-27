import torch
import librosa
import numpy as np
import torch.nn.functional as F
import torchaudio.functional as F_audio
from tqdm import tqdm

## Load WAV using librosa (avoids torchaudio backend issues)
def load_wav(path):
    x, sr = librosa.load(path, sr=None, mono=True)
    x = torch.from_numpy(x).float()
    x = x / (x.abs().max() + 1e-8)
    return x, sr

## Load IR using librosa
def load_ir(path):
    ir, sr = librosa.load(path, sr=None, mono=True)
    ir = torch.from_numpy(ir).float()
    ir = ir / (ir.abs().max() + 1e-8)
    return ir

## Oversample
def oversample(x, factor):
    return F_audio.resample(x, 1, factor)

## Downsample
def downsample(x, factor):
    return F_audio.resample(x, factor, 1)

## Windowing (for CNN input)
def make_window(x, y, window):
    X, Y = [], []
    for i in tqdm(range(window, len(x)), desc="Creating training windows"):
        X.append(x[i - window:i])
        Y.append(y[i])
    return torch.stack(X), torch.tensor(Y)

## Pre-Emphasis Filter
def pre_emphasis(x, coeff=0.95):
    return x[1:] - coeff * x[:-1]

## Apply CAB IR
def apply_ir(signal, ir):
    """
    signal: [N]
    ir:     [K]
    returns: [N + K - 1]
    """
    print(">>> APPLY_IR FUNCTION CALLED <<<")
    print("signal len:", signal.shape[0], "ir len:", ir.shape[0])

    signal = signal.unsqueeze(0).unsqueeze(0)  # [1,1,N]
    ir = ir.unsqueeze(0).unsqueeze(0)           # [1,1,K]

    padding = ir.shape[-1] - 1

    out = F.conv1d(
        signal,
        ir,
        padding=padding
    )
    print("after conv len:", out.shape[-1])


    out = out[:len(signal)]

    return out.squeeze()

