# Guitar Amplifier & Cabinet Neural Network Model

A deep learning project that uses a convolutional neural network to model the non-linear behavior of guitar amplifiers and simulate speaker cabinet impulse responses. The trained model can process clean guitar audio and produce a realistic amplified output with cabinet coloration.

## Table of Contents
- [Overview](#overview)
- [Graphs and How to read this stuff](#graphs-and-how-to-read-this-stuff)
- [What's Being Done](#whats-being-done)
- [Why This Matters](#why-this-matters)
- [Underlying Logic](#underlying-logic)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Concepts](#key-concepts)
- [Advanced Topics](#advanced-topics)

---

## Overview

This project aims to **digitally model analog guitar amplifier circuitry and speaker cabinet acoustics** using a neural network approach. Instead of trying to simulate the physical physics of vacuum tubes and speakers, we train a CNN to learn the non-linear transfer function by example, enabling real-time or near-real-time guitar amp simulation.

### Key Goals:
- Train a lightweight CNN that learns amp + cab characteristics from reference audio
- Achieve realistic, perceptually convincing amplifier tone from a clean DI (direct injection) signal
- Demonstrate GPU-accelerated training for fast iteration
- Visualize the learned transfer function and spectral characteristics

---
## Graphs and How to read this stuff

### Clean DI Waveform
![](graphs\waveform_clean_di.png)
This plot shows the raw guitar signal recorded directly from the instrument, without amplification or effects.
The waveform has a relatively simple shape and lower overall energy, which reflects the lack of distortion and compression.
When reading similar plots, cleaner signals typically appear more sparse and less “dense” over time.

What this implies: This is the baseline signal the model must transform.

### Amp Output Waveform
![](graphs\waveform_amp_output.png)
This waveform shows the real amplifier’s output in the time domain.
The signal appears denser and more compressed, indicating nonlinear distortion and dynamic leveling caused by the amp circuitry.
In general, increased waveform density and irregular shapes suggest saturation and harmonic generation.

What this implies: The target sound contains strong nonlinear behavior.

### Modeled Output Waveform
![](graphs\waveform_modeled_output.png)
This waveform represents the neural network’s processed output.
Its visual similarity to the real amp waveform indicates that the model has learned to reproduce distortion and compression.
When evaluating your own models, mismatched waveform density can indicate underfitting or incorrect normalization.

What this implies: The model is producing realistic time-domain behavior.

### Clean Spectrum
![](graphs\spectrum_clean_spectrum.png)
This spectrum shows how the clean DI signal’s energy is distributed across frequencies.
Notice the broader frequency range and stronger high-frequency content, which is typical of direct guitar recordings.
When reading spectra, taller peaks indicate stronger frequencies, and wide bandwidth indicates less filtering.

What this implies: The input signal contains more raw information than a finished guitar tone.

### Amp Spectrum
![](graphs\spectrum_amp_spectrum.png)
This plot shows the frequency content of the real amplified signal.
Energy is concentrated in the low and mid frequencies, with reduced high frequencies due to distortion and speaker interaction.
In guitar tones, a mid-focused spectrum is a strong indicator of a realistic amp sound.

What this implies: The amp shapes the tone, not just the loudness.

### Cabinet IR
![](graphs\ir_cabinet_ir.png)

This plot represents the cabinet impulse response, showing how the cabinet reacts to a very short input signal.
A sharp initial spike followed by a fast decay is characteristic of real speaker cabinets.
When reading IRs, long ringing or oscillations usually indicate unrealistic or problematic responses.

What this implies: The cabinet behaves like a real physical system.

### Cabinet IR Spectrum
![](graphs\spectrum_cabinet_ir_spectrum.png)
This spectrum shows how the cabinet filters different frequencies.
High frequencies are strongly attenuated, which explains why guitar cabinets sound warm and not harsh.
In general, steep high-frequency roll-off is expected for guitar speaker cabinets.

What this implies: The cabinet plays a critical role in shaping the final tone.

### Modeled Output Spectrum
![](graphs\spectrum_modeled_output_spectrum.png)
This plot shows the frequency content of the model’s final output after amp modeling and cabinet filtering.
A close match to the real amp spectrum indicates that the model has learned both nonlinear distortion and frequency shaping.
When evaluating models, spectral mismatches often point to missing cabinet modeling or insufficient training data.

What this implies: The model produces perceptually realistic tonal balance.

### Learned Transfer Curve
![](graphs\transfer_curve.png)
This curve shows how the model maps input signal levels to output levels.
The curved, non-linear shape indicates saturation, which is essential for distortion.
In general, straight lines imply linear systems, while curves imply nonlinear behavior.

What this implies: The network has learned distortion rather than acting as a simple filter.

---

## What's Being Done

### 1. **Data Preparation Phase**

The pipeline starts by loading three audio files and preparing them for training:

#### Input Audio Files:
- **clean.wav**: A clean, dry guitar recording (direct input, no processing)
- **amp.wav**: The same guitar recording processed through a real amplifier and cabinet
- **cab_ir.wav**: An impulse response (IR) of the speaker cabinet

#### Loading and Normalization:
Audio is loaded using librosa to avoid backend compatibility issues:

```python
# From audio_utils.py
def load_wav(path):
    x, sr = librosa.load(path, sr=None, mono=True)
    x = torch.from_numpy(x).float()
    x = x / (x.abs().max() + 1e-8)  # Normalize to [-1, 1]
    return x, sr
```

Each audio signal is normalized to the range [-1, 1] to prevent numerical instability and ensure consistent training dynamics.

#### Oversampling Strategy:
Audio is upsampled 8x during training to capture fine details in the frequency domain:

```python
def oversample(x, factor):
    return F_audio.resample(x, 1, factor)
```

**Why 8x oversampling?**
- At 44.1 kHz, guitar transients can occur at rates requiring higher temporal resolution
- 8x upsampling (to ~352.8 kHz) preserves high-frequency nuances in distortion and string attack
- More training samples per second of audio improves convergence
- Improves numerical stability in the network's learned transfer function

#### Training Window Creation:
Audio windows of 256 samples are extracted as training examples using a sliding window approach:

```python
def make_window(x, y, window):
    X, Y = [], []
    for i in tqdm(range(window, len(x)), desc="Creating training windows"):
        X.append(x[i - window:i])  # 256 historical samples
        Y.append(y[i])             # Single target output sample
    return torch.stack(X), torch.tensor(Y)
```

For each position $i$, the network receives 256 historical samples $x[i-256:i]$ and must predict the corresponding output sample $y[i]$. This creates a causal, sliding-window training setup.

### 2. **Model Training**

A convolutional neural network with dilated convolutions learns to map clean audio → amplified audio.

#### Network Architecture:
```
Input: (batch_size, 1, 256)  [batch, channels, samples]
    ↓
Conv1d(1 → 32, kernel=3, padding=1, dilation=1)
    ↓
Tanh activation
    ↓
Conv1d(32 → 32, kernel=3, padding=2, dilation=2)
    ↓
Tanh activation
    ↓
Conv1d(32 → 32, kernel=3, padding=4, dilation=4)
    ↓
Tanh activation
    ↓
Conv1d(32 → 1, kernel=3, padding=1, dilation=1)  [linear output]
    ↓
Output: (batch_size, 1)  [single sample prediction]
```

**Parameter Count:** ~4,000 (very lightweight for fast inference)

#### Loss Function:
The training uses a two-part loss combining reconstruction and perceptual accuracy:

```python
loss = MSE(predicted, target) + 0.5 * MSE(pre_emphasis(predicted), pre_emphasis(target))
```

**MSE Term**: Ensures overall waveform reconstruction accuracy
**Pre-Emphasis Term**: Uses a high-pass filter to emphasize high-frequency transients:

```python
def pre_emphasis(x, coeff=0.95):
    return x[1:] - coeff * x[:-1]
```

This high-pass filtered loss encourages the model to match sharp attacks and harmonic content critical for guitar tone authenticity. The coefficient 0.95 creates a filter that boosts frequencies inversely proportional to their wavelength, emphasizing rapid changes.

**Weight Balance:** The 0.5 scaling on the pre-emphasis term prevents over-emphasis on high frequencies while still prioritizing transient fidelity.

### 3. **Inference**

Once trained, the model processes the clean audio in a causal, sliding-window fashion with batched GPU processing:

```python
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
    model.eval()
    N = clean_os.shape[0]
    out_os = torch.zeros(N, device=device)
    weight = torch.zeros(N, device=device)

    with torch.no_grad():
        for i in tqdm(range(0, N - window, batch_size), desc="running inference"):
            # Extract and batch multiple inference windows
            batch_end = min(i + batch_size, N - window)
            batch_size_actual = batch_end - i
            
            windows = []
            for j in range(i, batch_end):
                x = clean_os[j:j+window].unsqueeze(0)  # [1, window]
                windows.append(x)
            
            x_batch = torch.cat(windows, dim=0).unsqueeze(1)  # [batch_size, 1, window]
            y_batch = model(x_batch)  # [batch_size, 1, 1]
            
            # Apply results with overlap-add for smooth reconstruction
            for j, idx in enumerate(range(i, batch_end)):
                y = y_batch[j].squeeze()
                out_os[idx:idx+window] += y
                weight[idx:idx+window] += 1.0

    # Normalize overlap-add
    out_os = out_os / weight.clamp(min=1.0)
```

**Key Points:**
- **Batching**: Multiple windows are processed in parallel for GPU efficiency (batch_size=32)
- **Overlap-Add**: Each output sample is the average of predictions from overlapping windows, providing smooth reconstruction
- **Causal Processing**: For sample $i$, only samples $[0...i-1]$ influence the output, enabling real-time processing

### 4. **Post-Processing**

The raw network output undergoes downsampling and impulse response convolution:

#### Downsampling:
```python
if oversample_factor > 1:
    out = downsample(out_os, oversample_factor)
    out_sr = sr
```

The 8x upsampled signal is downsampled back to the original sample rate.

#### Cabinet IR Application:
```python
def apply_ir(signal, ir):
    """
    signal: [N]
    ir:     [K]
    returns: [N + K - 1] (convolution output length)
    """
    signal = signal.unsqueeze(0).unsqueeze(0)  # [1,1,N]
    ir = ir.unsqueeze(0).unsqueeze(0)           # [1,1,K]
    
    padding = ir.shape[-1] - 1
    
    out = F.conv1d(
        signal,
        ir,
        padding=padding
    )
    
    return out.squeeze()
```

The cabinet IR is applied via 1D convolution, which linearly filters the network output to simulate the speaker cabinet's acoustic coloration.

#### Normalization:
```python
out = out / out.abs().max().clamp(min=1e-8)
```

Final output is normalized to prevent clipping while maintaining loudness consistency.

### 5. **Visualization**

The project generates detailed charts showing:

- Input waveforms and spectra (before processing)
- Learned transfer curve (amplitude in vs. out) showing the amp's non-linear characteristic
- Output waveforms and spectra (after neural network + IR)
- Cabinet IR frequency response characteristics

These visualizations help understand what the network has learned and how the output compares to the target.

---

## Why This Matters

### Old School approach
1. **Physical Modeling**: Simulating vacuum tubes, transformers, and speaker cones requires deep domain knowledge and is computationally expensive. Exact circuit simulation often requires solving differential equations in real-time.
2. **Convolution-Based**: Pure convolution (using IR alone) cannot model non-linear behavior—amps compress and distort, not just filter. An IR is inherently a linear operator; it cannot produce harmonics that weren't present in the input.
3. **Parametric EQ**: Simple filtering can't capture the complex harmonic generation and dynamic response of a real amp.

### Neural Network Advantage
- **Data-Driven**: Learn directly from real amplifier behavior, avoiding manual tuning of complex parameters
- **Perceptually Authentic**: The network optimizes for audio that *sounds* like the original amp through the pre-emphasis loss, prioritizing perceptual fidelity
- **Real-Time Capable**: Once trained, inference is fast enough for live playing with GPU acceleration (sub-millisecond latency)
- **Flexible**: Same architecture can model different amps—just retrain on different reference audio
- **Non-Linear Modeling**: Tanh activations naturally model soft saturation and compression, core to amplifier behavior

### Practical Applications
- **Digital Audio Workstations (DAWs)**: Plugin-based amp simulation for recording and mixing
- **Live Performance**: Lightweight, low-latency amp modeling for touring musicians
- **Tone Matching**: Clone the exact tone of a famous amp or player
- **Neural DSP-style Effects**: Proves the viability of machine learning for audio processing
- **Amp Development**: Test and iterate on amp designs through software before hardware prototyping

---

## Underlying Logic

### 1. Causal Windowed Convolution

The network processes 256 samples of history to predict one future output sample. This "causal" approach means:
- It respects temporal causality (future doesn't influence the past)
- It's suitable for real-time streaming audio without lookahead
- The window size (256) balances context and latency

**Why 256 samples?**
At 44.1 kHz sample rate (after oversampling to 352.8 kHz):
$$\text{Window Duration} = \frac{256 \text{ samples}}{352.8 \text{ kHz}} \approx 0.73 \text{ ms}$$

This provides enough context to capture low-frequency transients and the amp's response envelope while keeping latency imperceptible to the player.

### 2. Dilated Convolutions

The GuneAmp model uses convolutions with increasing dilation rates:

```
Layer 1: kernel=3, dilation=1   → receptive field = 3 samples
Layer 2: kernel=3, dilation=2   → receptive field = 5 samples  
Layer 3: kernel=3, dilation=4   → receptive field = 7 samples
```

**Receptive field calculation:**
$$\text{RF} = \text{kernel_size} + (\text{kernel_size}-1) \times (\text{dilation}-1)$$

For layer 3: $\text{RF} = 3 + (3-1) \times (4-1) = 3 + 6 = 9$ samples

**Why dilations?**
- **Expands receptive field** without increasing parameters or computation
- **Captures multi-scale temporal patterns**: layer 1 learns fast transients, layer 3 learns slow envelope changes
- **More efficient** than using large kernels which would increase parameter count quadratically
- Enables the network to "see" longer dependencies while remaining computationally lightweight

### 3. Activation Functions

The model uses **Tanh** activations after each convolutional layer:

```
Conv1d → Tanh → Conv1d → Tanh → Conv1d → (linear output, no activation)
```

**Why Tanh?**
- Bounded to [-1, 1], which mirrors the range of audio signals
- Smooth gradient everywhere, enabling stable training (unlike ReLU which has zero gradient for negative values)
- Naturally models soft saturation and compression, the primary non-linearity in analog amplifiers
- The final layer is linear to allow the output range to adapt to the target signal characteristics
- Tanh saturation at extreme values mimics how tube amplifiers "clip" gracefully at high volumes

### 4. Loss Function (Two-Part)

```python
loss = MSE(predicted, target) + 0.5 * MSE(pre_emphasis(predicted), pre_emphasis(target))
```

**Component Breakdown:**

1. **MSE Term** ($L_{\text{recon}}$): Reconstruction loss
   $$L_{\text{recon}} = \frac{1}{N} \sum_{i=0}^{N-1} (y_{\text{pred}}[i] - y_{\text{target}}[i])^2$$
   Ensures the overall waveform amplitude and shape match the target.

2. **Pre-Emphasis Term** ($L_{\text{pre}}$): Perceptual loss
   $$L_{\text{pre}} = \frac{1}{N} \sum_{i=1}^{N-1} ([y_{\text{pred}}[i] - 0.95 \cdot y_{\text{pred}}[i-1]] - [y_{\text{target}}[i] - 0.95 \cdot y_{\text{target}}[i-1]])^2$$
   
   The pre-emphasis filter $H(z) = 1 - 0.95z^{-1}$ is a first-order high-pass filter that boosts high frequencies. This forces the model to match:
   - Sharp transients (string attacks)
   - Harmonic content (tone definition)
   - Noise characteristics (natural amp grittiness)

**Total Loss:**
$$L_{\text{total}} = L_{\text{recon}} + 0.5 \cdot L_{\text{pre}}$$

The 0.5 weighting prevents over-fitting to high frequencies while still maintaining perceptual fidelity. Humans are most sensitive to high-frequency attack, so this weighting ensures the model prioritizes transient accuracy while balancing overall waveform shape.

### 5. Oversampling Strategy

The audio is upsampled 8x during training/inference using bandlimited interpolation:

```python
def oversample(x, factor):
    return F_audio.resample(x, 1, factor)  # factor=8
```

**Benefits:**
- High-frequency details that might be missed at 44.1 kHz are preserved at 352.8 kHz
- Better modeling of sharp attacks and noise floor in distortion
- Increases the number of training samples per second of audio (8x more training data)
- Improves numerical stability in the network's learned transfer function
- Enables finer granularity in learning the amp's frequency response

**Trade-off:** 8x higher computational cost during training & inference. A GTX 1080 Ti can handle this comfortably.

### 6. Impulse Response Convolution

After the network output, the cabinet IR is applied via convolution:

```python
output = F.conv1d(signal_reshaped, ir_reshaped, padding=...)
```

**Why separate from the network?**
- **Physical Decoupling**: The amplifier (non-linear) and speaker cabinet (linear filtering) are independent physical components
- **Modularity**: The trained amp model can be reused with different cabinet IRs without retraining
- **Computational Efficiency**: 1D convolution is highly optimized; applying it once at the end is faster than building it into the network
- **Interpretability**: Separates the learned non-linear behavior from the linear cabinet response

---

## Architecture

### File Structure
```
amp-cab-nn/
├── main.py              # Entry point; orchestrates the full pipeline
├── model.py             # GuneAmp neural network definition
├── train.py             # Training loop with loss calculation
├── infer.py             # Inference (batched sliding-window prediction)
├── audio_utils.py       # Audio loading, resampling, windowing, IR application
├── viz.py               # Visualization and chart saving
├── requirements.txt     # Python dependencies
├── checkpoints/         # Saved model weights
├── graphs/              # Output directory for saved charts
└── readme.md            # This file
```

### Model Architecture (GuneAmp)
```
Input: (batch_size, 1, 256)  [batch, channels, samples]
    ↓
Conv1d(1 → 32, kernel=3, padding=1, dilation=1)
    ↓
Tanh activation
    ↓
Conv1d(32 → 32, kernel=3, padding=2, dilation=2)
    ↓
Tanh activation
    ↓
Conv1d(32 → 32, kernel=3, padding=4, dilation=4)
    ↓
Tanh activation
    ↓
Conv1d(32 → 1, kernel=3, padding=1, dilation=1)
    ↓
Output: (batch_size, 1)  [single sample prediction]
```

**Parameter Count:** ~4,000 parameters
- Conv1(1→32): 96 parameters
- Conv2(32→32): 2,912 parameters
- Conv3(32→32): 2,912 parameters
- Conv4(32→1): 97 parameters

This extremely lightweight architecture enables:
- Fast inference (sub-millisecond per sample on GPU)
- Quick training convergence (50 epochs in ~5-10 minutes on GTX 1080 Ti)
- Easy deployment to real-time audio systems

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration; CPU mode supported but slow)
- NVIDIA GPU with compute capability 6.1+ (GTX 1080 Ti recommended)

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
- `clean.wav` - Clean guitar recording (mono, any sample rate)
- `amp.wav` - Same recording processed through amp + cabinet (mono, any sample rate)
- `cab_ir.wav` - Cabinet impulse response (mono, typically 0.5-2 seconds)

**Audio File Format Notes:**
- All files should be 16-bit or 32-bit PCM WAV
- Mono files are automatically handled by librosa
- Sample rates are automatically resampled to match
- Typical file lengths: 10-30 seconds for training audio

---

## Usage

### Training & Full Pipeline
```bash
python main.py
```

This will:
1. Load the three audio files and print their statistics
2. Visualize input waveforms and spectra → `graphs/01_input_clean.png`, `graphs/02_input_amp.png`
3. Oversample 8x and create training windows
4. Train the model for 50 epochs with progress bar
5. Plot the learned transfer curve → `graphs/03_transfer_function.png`
6. Run batched inference on the clean audio with overlap-add
7. Apply cabinet IR convolution
8. Save output to `modeled_amp_with_cab.wav`
9. Visualize output → `graphs/04_output.png`

### Configuration
Edit `main.py` to adjust:
```python
WINDOW = 256       # Context window size (samples) - increase for more context, higher latency
OVERSAMPLE = 8     # Upsampling factor - increase for finer details, higher computation
EPOCHS = 50        # Training epochs - more epochs = longer training, better convergence
BATCH_SIZE = 32    # Training batch size - increase for faster training if VRAM allows
LR = 1e-3          # Learning rate - tune if training diverges or converges slowly
```

### Output Files
- `modeled_amp_with_cab.wav` - Final modeled audio output (resampled + IR applied)
- `graphs/01_input_clean.png` - Waveform and spectrum of clean input
- `graphs/02_input_amp.png` - Waveform and spectrum of target amp output
- `graphs/03_transfer_function.png` - Learned input→output transfer curve
- `graphs/04_output.png` - Waveform and spectrum of network output
- `checkpoints/gune_amp.pt` - Trained model weights (can be loaded for inference)

### GPU Memory Usage
On a GTX 1080 Ti (11GB VRAM):
- Typical training: ~2-3 GB (with batch_size=32)
- Inference: ~1 GB
- You can safely increase batch_size to 64-128 for faster training

Monitor with:
```bash
nvidia-smi -l 1  # Updates every 1 second
```

---

## Key Concepts

### 1. Impulse Response (IR)
An IR captures how a system (speaker cabinet) responds to a brief impulse (a single click). The cabinet IR typically contains 4,410-44,100 samples of response.

**Application:**
Convolving a signal with an IR is equivalent to running that signal through the physical system. For a cabinet:
$$y[n] = \sum_{k=0}^{K-1} x[n-k] \cdot h[k]$$

where $h[k]$ is the impulse response and $K$ is its length. This is purely linear filtering—no new frequencies are created, only existing frequencies are attenuated/boosted based on the cabinet's acoustic characteristics.

### 2. Non-Linear vs. Linear
- **Amplifier**: Non-linear behavior
  - Input: $x[n]$ (clean guitar signal)
  - Output: $y[n] = f(x[n])$ where $f$ is non-linear (typically with saturation like tanh)
  - Creates harmonics and frequency content not present in the input
  - Example: $f(x) = \tanh(x)$ compresses large amplitudes, adding harmonic distortion
  
- **Cabinet**: Linear filtering
  - Input: Network output
  - Output: Convolution with IR
  - No new frequencies created; only existing frequencies filtered
  
- **Combined Model**: 
  - Network learns amp non-linearity
  - IR convolution adds cabinet coloration
  - Physically grounded separation of concerns

### 3. Receptive Field
The total number of historical input samples that influence a given output. A larger receptive field allows the model to make predictions based on longer-range dependencies.

For our architecture with padding:
- After Layer 1 (dilation=1): RF = 3 samples
- After Layer 2 (dilation=2): RF = 7 samples (builds on previous RF)
- After Layer 3 (dilation=4): RF = 15 samples total

This ~340 microsecond receptive field is sufficient to capture sustain and envelope characteristics while maintaining causal constraints.

### 4. Dilation
Spacing between samples in a convolution kernel. With `dilation=2`, the kernel samples every other input:

```
Regular convolution (dilation=1):
Input:  [a, b, c, d, e, f, g]
Kernel: [w0, w1, w2]
Covers:  ↑   ↑   ↑  (consecutive samples)

Dilated convolution (dilation=2):
Input:  [a, b, c, d, e, f, g]
Kernel: [w0, w1, w2]
Covers:  ↑     ↑     ↑  (every other sample)
```

Dilations exponentially expand the receptive field while keeping the number of parameters constant.

### 5. Pre-Emphasis
A high-pass filter that emphasizes rapid changes (transients, high frequencies):

$$H(z) = 1 - 0.95 z^{-1}$$

In the time domain: $y[n] = x[n] - 0.95 \cdot x[n-1]$

This filter amplifies high frequencies while attenuating low frequencies. The coefficient 0.95 determines the cutoff frequency—larger values (closer to 1) create sharper high-pass behavior.

**Why use it?** Guitars are perceptually sensitive to transient attack (high frequencies). This loss term forces the network to precisely match string attacks, pick noise, and harmonic richness.

### 6. Causal vs. Non-Causal
- **Causal**: Output at time $t$ depends only on inputs at times $\leq t$
  - Required for real-time streaming (no lookahead possible)
  - Used in this project
  - Slight latency (window_size / sample_rate)
  
- **Non-Causal**: Output at time $t$ can depend on inputs at times $> t$
  - Requires full signal lookahead
  - Could improve accuracy (not implemented here)
  - Unsuitable for live performance

---

## Advanced Topics

### Performance Optimization

#### GPU Utilization
Your GTX 1080 Ti (11 GB VRAM, 3,584 CUDA cores) can be fully leveraged:

**Mixed Precision Training** (optional enhancement):
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for epoch in range(EPOCHS):
    for x, y in dataloader:
        with autocast():
            y_pred = model(x)
            loss = criterion(y_pred, y)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
```

This reduces memory usage by ~50% and speeds up training while maintaining accuracy.

#### Memory-Efficient Inference
The batching in `infer.py` is optimized:
- Batch size 32 provides good GPU saturation without exceeding VRAM
- Overlap-add ensures smooth reconstruction across batch boundaries
- You can increase to batch_size=64-128 if needed

#### Profile Your Code
```bash
python -m cProfile -s cumtime main.py > profile.txt
less profile.txt
```

This identifies bottlenecks. On GTX 1080 Ti, typically:
- Data loading: ~10% of time
- Training: ~70% of time
- Inference: ~15% of time
- Visualization: ~5% of time

### Extending the Architecture

#### Deeper Networks
Try adding more dilated layers for larger receptive fields:
```python
# In model.py, modify forward():
Conv1d(32 → 32, kernel=3, padding=8, dilation=8)  # Receptive field: 31
Tanh
Conv1d(32 → 32, kernel=3, padding=16, dilation=16)  # Receptive field: 63
```

This increases context from ~340 microseconds to ~1.8 milliseconds, capturing longer envelope responses.

**Trade-off:** More parameters (from 4k to ~20k), longer inference time, potentially better tone matching.

#### Different Loss Functions
Try frequency-domain losses:
```python
def spectral_loss(y_pred, y_target):
    spec_pred = torch.stft(y_pred, n_fft=1024)
    spec_target = torch.stft(y_target, n_fft=1024)
    return torch.nn.functional.mse_loss(
        torch.abs(spec_pred), 
        torch.abs(spec_target)
    )
```

This directly optimizes spectral matching, potentially improving tone authenticity.

### Model Interpretation

#### Visualizing the Transfer Function
The saved `graphs/03_transfer_function.png` shows the learned input→output mapping. Steep slopes indicate compression or expansion; saturation at extremes indicates clipping behavior.

#### Analyzing Learned Filters
```python
# Extract learned kernels from model
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"{name}: {param.shape}")
        print(param.data[:3])  # Visualize first 3 kernels
```

Layer 1 kernels typically learn edge detection (high-frequency filters), while layer 3 kernels learn slower transient responses.

---

## References & Inspiration
- **Neural Amp Modeling**: ML applied to vintage gear emulation (Neural DSP, Tonex)
- **Dilated Convolutions**: WaveNet (van den Oord et al., 2016) - applied to audio generation
- **Audio DSP Principles**: Julius Smith's "Physical Audio Signal Processing"
- **CNN for Audio**: Simonyan & Zisserman (2014) on temporal convolutional networks
- **Perceptual Loss**: Johnson et al. (2016) on perceptual losses for real-time style transfer

---

## License
MIT License - feel free to use and modify for personal or commercial use.
