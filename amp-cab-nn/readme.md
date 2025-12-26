# Guitar Amplifier & Cabinet Neural Network Model

A deep learning project that uses a convolutional neural network to model the non-linear behavior of guitar amplifiers and simulate speaker cabinet impulse responses. The trained model can process clean guitar audio and produce a realistic amplified output with cabinet coloration.

## Table of Contents
- [Overview](#overview)
- [What's Being Done](#whats-being-done)
- [Why This Matters](#why-this-matters)
- [Underlying Logic](#underlying-logic)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Concepts](#key-concepts)

---

## Overview

This project aims to **digitally model analog guitar amplifier circuitry and speaker cabinet acoustics** using a neural network approach. Instead of trying to simulate the physical physics of vacuum tubes and speakers, we train a CNN to learn the non-linear transfer function by example, enabling real-time or near-real-time guitar amp simulation.

### Key Goals:
- Train a lightweight CNN that learns amp + cab characteristics from reference audio
- Achieve realistic, perceptually convincing amplifier tone from a clean DI (direct injection) signal
- Demonstrate GPU-accelerated training for fast iteration
- Visualize the learned transfer function and spectral characteristics

---

## What's Being Done

### 1. **Data Preparation Phase**
The pipeline starts by loading three audio files:

- **clean.wav**: A clean, dry guitar recording (direct input, no processing)
- **amp.wav**: The same guitar recording processed through a real amplifier and cabinet
- **cab_ir.wav**: An impulse response (IR) of the speaker cabinet

These are normalized and prepared for training:
- Audio is resampled/oversampled 8x to capture fine details in the frequency domain
- Audio windows of 256 samples are extracted as training examples
- Oversampling allows the network to learn high-frequency nuances

### 2. **Model Training**
A convolutional neural network with dilated convolutions learns to map clean audio → amplified audio:

- **Input**: 256-sample windows of clean guitar audio
- **Processing**: 3 dilated convolutional layers with increasing dilation rates (1, 2, 4)
- **Output**: Single predicted sample (the network learns to predict one output sample at a time)
- **Loss Function**: Combines reconstruction loss (MSE) + perceptual loss (pre-emphasized MSE)

The pre-emphasis loss encourages the model to match high-frequency transients, which are critical for guitar tone authenticity.

### 3. **Inference**
Once trained, the model processes the clean audio in a causal, sliding-window fashion:

- For each sample position, we extract a 256-sample window from the past
- Pass it through the network
- Capture the predicted output
- Repeat for the entire audio signal

### 4. **Post-Processing**
The raw network output is further processed:

- **Downsampled** back to original sample rate (inverse of the 8x oversampling)
- **Cabinet IR applied** via convolution to add speaker/room coloration
- **Normalized** to prevent clipping

### 5. **Visualization**
The project generates and saves detailed charts showing:

- Input waveforms and spectra
- Learned transfer curve (amplitude in vs. out)
- Output waveforms and spectra
- Cabinet IR characteristics

---

## Why This Matters

### Traditional Approaches (Limitations)
1. **Physical Modeling**: Simulating vacuum tubes, transformers, and speaker cones requires deep domain knowledge and is computationally expensive
2. **Convolution-Based**: Pure convolution (using IR alone) cannot model non-linear behavior—amps compress and distort, not just filter
3. **Parametric EQ**: Simple filtering can't capture the complex harmonic generation of a real amp

### Neural Network Advantage
- **Data-Driven**: Learn directly from real amplifier behavior, avoiding manual tuning
- **Perceptually Authentic**: The network optimizes for audio that *sounds* like the original amp, not just mathematical accuracy
- **Real-Time Capable**: Once trained, inference is fast enough for live playing (with GPU acceleration)
- **Flexible**: Same architecture can model different amps—just retrain on different reference audio

### Practical Applications
- **Digital Audio Workstations (DAWs)**: Plugin-based amp simulation for recording
- **Live Performance**: Lightweight, low-latency amp modeling for touring musicians
- **Tone Matching**: Clone the exact tone of a famous amp or player
- **Neural DSP-style Effects**: Proves the viability of machine learning for audio processing

---

## Underlying Logic

### 1. Causal Windowed Convolution
The network processes 256 samples of history to predict one future output sample. This "causal" approach means:
- It respects the temporal causality (future doesn't influence the past)
- It's suitable for real-time streaming audio
- The window size (256) is a balance between context and latency

**Why 256 samples?**
At typical 44.1 kHz sample rate, 256 samples ≈ 5.8 ms of history. This is enough to capture low-frequency transients while keeping latency imperceptible.

### 2. Dilated Convolutions
The GuneAmp model uses convolutions with increasing dilation:

```
Layer 1: kernel_size=3, dilation=1   → receptive field = 3 samples
Layer 2: kernel_size=3, dilation=2   → receptive field = 5 samples  
Layer 3: kernel_size=3, dilation=4   → receptive field = 7 samples
```

**Why dilations?**
- **Expands receptive field** without increasing parameters or computation
- **Captures multi-scale temporal patterns**: layer 1 learns fast transients, layer 3 learns slow envelope changes
- **More efficient** than using large kernels

### 3. Activation Functions
The model uses **Tanh** activations after each conv layer:

```python
Conv1d → Tanh → Conv1d → Tanh → Conv1d → (no activation, linear output)
```

**Why Tanh?**
- Bounded to [-1, 1], which mirrors audio signal behavior
- Smooth gradient, better for training
- Naturally models soft saturation/compression common in amplifiers
- The final layer is linear to allow output range to match target signals

### 4. Loss Function (Two-Part)
```python
loss = MSE(predicted, target) + 0.5 * MSE(pre_emphasis(predicted), pre_emphasis(target))
```

**Why two terms?**
1. **MSE Term**: Ensures overall waveform accuracy
2. **Pre-Emphasis Term**: 
   - Pre-emphasis filter boosts high frequencies: `y[n] = x[n] - 0.95 * x[n-1]`
   - Forces the model to match high-frequency transients and peaks
   - Humans are sensitive to high-frequency attack—this makes the model prioritize them
   - 0.5 weight prevents over-fitting to high frequencies

### 5. Oversampling Strategy
The audio is upsampled 8x during training/inference:

**Benefits:**
- High-frequency details that might be missed at 44.1 kHz are preserved
- Better modeling of sharp attacks and noise in distortion
- More training samples per second of audio
- Improves numerical stability in the network

**Trade-off:** 8x higher computational cost during training & inference

### 6. Impulse Response Convolution
After the network output, the cabinet IR is applied via convolution:

```python
output = conv1d(network_output, cab_ir)
```

**Why separate from the network?**
- The amp (non-linear behavior) and cabinet (linear filtering) are physically decoupled
- Separating them allows reusing the same amp model with different IRs
- Computational efficiency: convolution is fast, and we only do it once at the end

---

## Architecture

### File Structure
```
amp-cab-nn/
├── main.py              # Entry point; orchestrates the full pipeline
├── model.py             # GuneAmp neural network definition
├── train.py             # Training loop with loss calculation
├── infer.py             # Inference (sliding-window prediction)
├── audio_utils.py       # Audio loading, resampling, windowing, IR application
├── viz.py               # Visualization and chart saving
├── requirements.txt     # Python dependencies
├── graphs/              # Output directory for saved charts
└── readme.md            # This file
```

### Model Architecture (GuneAmp)
```
Input: (batch_size, 1, 256)  [batch, channels, samples]
    ↓
Conv1d(1 → 32, kernel=3, padding=1, dilation=1)  [receptive field: 3]
    ↓
Tanh activation
    ↓
Conv1d(32 → 32, kernel=3, padding=2, dilation=2) [receptive field: 5]
    ↓
Tanh activation
    ↓
Conv1d(32 → 32, kernel=3, padding=4, dilation=4) [receptive field: 7]
    ↓
Tanh activation
    ↓
Conv1d(32 → 1, kernel=3, padding=1, dilation=1)  [final linear layer]
    ↓
Output: (batch_size, 1)  [single sample prediction]
```

**Parameter Count:** ~4,000 (very lightweight)

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration; CPU mode supported but slow)

### Setup
```bash
# Clone or navigate to the project
cd amp-cab-nn

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For CUDA-enabled PyTorch (recommended for performance):
pip install --upgrade "torch" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Audio Files Required
Place these files in the project root:
- `clean.wav` - Clean guitar recording
- `amp.wav` - Same recording processed through amp + cabinet
- `cab_ir.wav` - Cabinet impulse response

---

## Usage

### Training & Full Pipeline
```bash
python main.py
```

This will:
1. Load the three audio files
2. Visualize input waveforms and spectra → `graphs/`
3. Oversample 8x and create training windows
4. Train the model for 50 epochs (adjustable in `main.py`)
5. Plot the learned transfer curve → `graphs/`
6. Run inference on the clean audio
7. Apply cabinet IR convolution
8. Save output to `modeled_amp_with_cab.wav`
9. Visualize output → `graphs/`

### Configuration
Edit `main.py` to adjust:
```python
WINDOW = 256       # Context window size (samples)
OVERSAMPLE = 8     # Upsampling factor
EPOCHS = 50        # Training epochs
```

### Output Files
- `modeled_amp_with_cab.wav` - Final modeled audio output
- `graphs/*.png` - All generated visualizations

---

## Key Concepts

### 1. Impulse Response (IR)
An IR captures how a system (speaker cabinet) responds to a brief impulse (click). Applying an IR via convolution is a linear operation that simulates the cabinet's filtering.

### 2. Non-Linear vs. Linear
- **Amplifier**: Non-linear (tanh activations simulate saturation/compression)
- **Cabinet**: Linear (modeled as convolution with IR)
- **Combined**: The network learns amp behavior; IR adds cabinet coloration

### 3. Receptive Field
The total history "visible" to the final layer. Larger receptive fields allow the model to consider longer-range dependencies (e.g., the amp's response to a sustained note).

### 4. Dilation
Spacing between samples in a convolution kernel. `dilation=2` means skip every other sample, expanding the field of view without increasing parameters.

### 5. Pre-Emphasis
A high-pass filter that emphasizes rapid changes (transients, high frequencies). It's applied to the loss to guide the model toward realistic attack and harmonic content.

### 6. Causal vs. Non-Causal
- **Causal**: Output depends only on current + past inputs (required for real-time streaming)
- **Non-Causal**: Output can depend on future samples (not used here, but valid for offline processing)

---

## Performance Notes

### Hardware Recommendations
- **GPU (Recommended)**: NVIDIA GTX 1080 Ti or better
  - Training: ~5-10 minutes for 50 epochs on typical audio
  - Inference: Real-time or faster
  
- **CPU**: Possible but slow
  - Training: 30+ minutes
  - Inference: May drop frames in live scenario

### Optimization Tips
1. **Reduce OVERSAMPLE** if speed is critical (trade-off: lower fidelity)
2. **Reduce WINDOW** to reduce memory (trade-off: less context)
3. **Use smaller audio files** for faster testing during development
4. **GPU Monitoring**: Watch GPU memory usage with `nvidia-smi`

---

## Future Improvements
- [ ] Export to plugin format (VST, AU) for DAW integration
- [ ] Train on multiple amp/cab combinations and allow selection
- [ ] Add pre-trained model weights for quick inference
- [ ] Implement non-causal mode for offline "perfect" modeling
- [ ] Add frequency-domain loss terms for better spectral matching
- [ ] Interactive real-time tone shaping parameter controls

---

## References & Inspiration
- Neural amp modeling: ML applied to vintage gear emulation
- Dilated convolutions: WaveNet (van den Oord et al., 2016)
- Audio DSP principles: Julius Smith's "Physical Audio Signal Processing"

---

## License
MIT License - feel free to use and modify for personal or commercial use.

---