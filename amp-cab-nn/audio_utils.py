import torch
import librosa
import numpy as np
import torchaudio.functional as F
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
    return F.resample(x, 1, factor)

## Downsample
def downsample(x, factor):
    return F.resample(x, factor, 1)

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
    signal = signal.unsqueeze(0).unsqueeze(0)
    ir = ir.flip(0).unsqueeze(0).unsqueeze(0)
    out = torch.nn.functional.conv1d(
        signal,
        ir,
        padding=ir.shape[-1] - 1
    )
    return out.squeeze()