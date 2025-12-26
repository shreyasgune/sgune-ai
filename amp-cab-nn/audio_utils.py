import torch
import torchaudio
import torchaudio.functional as F

## Load WAV
def load_wav(path):
    x, sr = torchaudio.load(path)
    x = x.mean(dim=0)
    x = x / x.abs().max()
    return x, sr 

## Load IR
def load_ir(path):
    ir, _ = torchaudio.load(path)
    ir  = ir.mean(dim=0)
    ir = ir / ir.abs().max()
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
    for i in range(window, len(x)):
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