import torch
from tqdm import tqdm
from audio_utils import load_wav, load_ir, oversample, make_window
from train import train_model
from infer import run_inference
# from torchcodec.decoders import AudioDecoder
from viz import plot_waveform, plot_spectrum, plot_ir, plot_transfer_curve
import os
from datetime import datetime

# Gotta use GPU if available 
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("USING DEVICE: ", DEVICE)
if DEVICE == "cuda":
    try:
        torch.cuda.set_device(0)
        print("GPU:", torch.cuda.get_device_name(0))
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True
print()

# DAS SETTINGS
WINDOW = 256
OVERSAMPLE = 1
EPOCHS = 10
BATCH_SIZE = 256
LEARN_RATE = 2e-5

#Load Audio
print("Loading audio files...")
clean, sr = load_wav("untitled.wav")
amp, _ = load_wav("amp.wav")
cab_ir = load_ir("cab_ir.wav")
print("Audio files loaded!\n")

clean *= 0.9
amp *= 0.9

print(clean.max())
print(amp.max())


#Oversample (for future anti-aliasing stuff)
# print("Oversampling audio...")
# clean_os = oversample(clean, OVERSAMPLE)
# amp_os = oversample(amp, OVERSAMPLE)
# print("Oversampling complete!\n")

clean_os = clean 
amp_os = amp
print("Preparing training data...")
X, Y = make_window(clean_os, amp_os, WINDOW)
X = X.unsqueeze(1)
Y = Y.unsqueeze(1).unsqueeze(2)
print(f"Training data prepared: {X.shape} samples\n")

# DAS TRAIN
model = train_model(X, Y, DEVICE, epochs=EPOCHS, batch_size=BATCH_SIZE, learn_rate=LEARN_RATE, window=WINDOW, checkpoint_path="checkpoints/gune_amp.pt", resume=True)



# Inference
print("Running inference...")

output = run_inference(
    model,
    clean_os,
    cab_ir,
    OVERSAMPLE,
    WINDOW,
    DEVICE,
    sr,
    output_file="modeled_amp_with_cab.wav"
)
print()

print("Visualizing input audio...")
plot_waveform(clean, sr, "CLEAN DI")
plot_waveform(amp, sr, "Amp Output")
plot_spectrum(clean, sr, "Clean Spectrum")
plot_spectrum(amp, sr, "Amp Spectrum")
plot_ir(cab_ir, sr, "Cabinet IR")
plot_spectrum(cab_ir, sr, "Cabinet IR Spectrum")
print("Generating transfer curve...")
plot_transfer_curve(model, WINDOW, DEVICE)
print("Transfer curve complete!\n")
print("Input visualizations complete!\n")

# Viz the output
print("Visualizing output...")
plot_waveform(output, sr, "Modeled Output")
plot_spectrum(output, sr, "Modeled Output Spectrum")
print("\nAll done!")
