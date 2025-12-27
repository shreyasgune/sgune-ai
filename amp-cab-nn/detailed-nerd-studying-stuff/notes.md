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
