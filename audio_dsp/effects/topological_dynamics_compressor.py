import numpy as np
from audio_dsp.utils import wav_io as wavfile
import scipy.signal
import matplotlib.pyplot as plt
import wave
import struct

def topological_compressor(input_file, output_file, threshold=-20, ratio=4.0, persistence_scale=0.1):
    """
    Dynamics compressor approximating topological features without persistent homology.
    
    Parameters:
    - input_file: Input WAV file
    - output_file: Output WAV file
    - threshold: dB threshold (default -20)
    - ratio: Compression ratio (default 4.0)
    - persistence_scale: Minimum prominence for significant peaks (default 0.1)
    """
    # Read WAV
    sample_rate, data = wavfile.read(input_file)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    data = data.astype(float) / np.iinfo(data.dtype).max

    # Phase space embedding
    derivative = np.diff(data, prepend=data[0]) * sample_rate
    point_cloud = np.vstack((data, derivative)).T

    # Approximate topological features with peak detection
    envelope = np.abs(data)
    peaks, properties = scipy.signal.find_peaks(envelope, height=persistence_scale, prominence=persistence_scale)
    prominences = properties['prominences']
    persistent_mask = prominences > persistence_scale  # Filter by prominence
    time_indices = peaks[persistent_mask]
    persistent_points = np.array([[data[i], 0] for i in time_indices])  # Mimic persistence diagram

    # Envelope based on persistent-like peaks
    smoothed_envelope = np.copy(envelope)
    for idx in time_indices:
        window = slice(max(0, idx-50), min(len(data), idx+50))
        smoothed_envelope[window] = np.maximum(smoothed_envelope[window], envelope[idx])

    # Smooth with attack/release
    attack = 0.01
    release = 0.1
    attack_coeff = np.exp(-1.0 / (attack * sample_rate))
    release_coeff = np.exp(-1.0 / (release * sample_rate))
    for i in range(1, len(smoothed_envelope)):
        coeff = attack_coeff if smoothed_envelope[i] > smoothed_envelope[i-1] else release_coeff
        smoothed_envelope[i] = coeff * smoothed_envelope[i-1] + (1 - coeff) * smoothed_envelope[i]

    # Gain reduction
    envelope_db = 20 * np.log10(np.maximum(smoothed_envelope, 1e-10))
    gain_db = np.zeros_like(envelope_db)
    above_threshold = envelope_db > threshold
    gain_db[above_threshold] = threshold + (envelope_db[above_threshold] - threshold) / ratio - envelope_db[above_threshold]
    gain_linear = 10 ** (gain_db / 20.0)

    # Apply gain
    output = data * gain_linear

    # Normalize and write WAV
    output = output / (np.max(np.abs(output)) * 1.1)
    output = (output * 32767).astype(np.int16)
    with wave.open(output_file, 'w') as wav_file:
        wav_file.setparams((1, 2, sample_rate, len(output), 'NONE', 'not compressed'))
        for sample in output:
            wav_file.writeframes(struct.pack('h', sample))

    # Visualization
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1. Phase Space Trajectory with Persistent-like Peaks
    axs[0].plot(point_cloud[:, 0], point_cloud[:, 1], 'b-', alpha=0.3, label='Trajectory')
    if len(persistent_points) > 0:
        axs[0].scatter(persistent_points[:, 0], np.zeros_like(persistent_points[:, 0]), c='red', label='Significant Peaks')
    axs[0].set_title("Phase Space Trajectory (Amplitude vs. Derivative)")
    axs[0].set_ylabel("Derivative")
    axs[0].legend()

    # 2. Original and Smoothed Envelope
    t = np.arange(len(data)) / sample_rate
    axs[1].plot(t, envelope, 'b-', alpha=0.5, label='Original Envelope')
    axs[1].plot(t, smoothed_envelope, 'r-', label='Pseudo-Topological Envelope')
    axs[1].set_title("Envelope with Significant Peak Smoothing")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()

    # 3. Gain Reduction and Output
    axs[2].plot(t, gain_db, 'g-', label='Gain Reduction (dB)')
    axs[2].plot(t, output / 32767.0, 'b-', alpha=0.5, label='Compressed Signal')
    axs[2].set_title("Gain Reduction and Compressed Output")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("dB / Amplitude")
    axs[2].legend()

    plt.tight_layout()
    plt.savefig("topological_compressor_visualization.png")
    plt.close()

def main():
    input_file = "input.wav"  # Replace with your WAV
    output_file = "topological_compressed.wav"
    print("Processing with topological dynamics compressor (workaround mode)...")
    topological_compressor(input_file, output_file, threshold=-20, ratio=4.0, persistence_scale=0.1)
    print(f"Created: {output_file}")
    print("Visualization saved as 'topological_compressor_visualization.png'")

if __name__ == "__main__":
    main()


    '''


## Topological Dynamics Compressor

The **Topological Dynamics Compressor** is an innovative audio processing algorithm that applies dynamic range compression using topological data analysis (TDA). Unlike traditional compressors that rely on amplitude thresholds or frequency-domain analysis, this method treats the audio signal’s amplitude trajectory as a geometric object in phase space, identifying and compressing only the most “significant” dynamic events based on their topological persistence. The result is a uniquely sculpted sound—preserving the signal’s natural contours while taming disruptive peaks, delivering a crystalline clarity.

### Concept
- **Phase Space Representation**: The audio signal is embedded into a 2D phase space, where each point represents the signal’s amplitude \( x(t) \) and its approximate derivative \( \dot{x}(t) \). This transforms the 1D time-domain signal into a trajectory that captures its dynamic behavior.
- **Persistent Homology**: Using TDA, the algorithm analyzes this trajectory as a point cloud, identifying features (e.g., peaks) that persist across scales. These persistent features correspond to significant dynamic events—loud transients or abrupt changes—that disrupt the signal’s “shape.”
- **Selective Compression**: Gain reduction is applied only to these topologically significant events, leaving less persistent (subtle) dynamics intact. This preserves the signal’s natural flow while sculpting its extremes.
- **Unique Sonic Twist**: The output has a polished, “crystalline” quality—major peaks are chiseled down, and the signal feels clarified, like audio carved into a precise, elegant form.

### Mathematical Foundation
1. **Phase Space Embedding**:
   - Define the point cloud as \( P = \{(x(t), \dot{x}(t))\} \), where:
     - \( x(t) \): Signal amplitude at time \( t \).
     - \( \dot{x}(t) \approx (x(t) - x(t-1)) \cdot f_s \): First derivative, approximated via finite difference with sample rate \( f_s \).
   - This creates a 2D trajectory reflecting amplitude and rate of change.

2. **Persistent Homology**:
   - **Vietoris-Rips Filtration**: Applied to the point cloud \( P \) using the `ripser` library, computing 0-dimensional (0D) homology—connected components (peaks).
   - **Persistence Diagram**: Outputs pairs \( (b_i, d_i) \), where \( b_i \) is the “birth” (appearance) and \( d_i \) is the “death” (merging) of a feature as the filtration scale increases.
   - **Persistence**: \( p_i = d_i - b_i \), measuring how long a feature persists. High-persistence points (e.g., \( p_i > 0.1 \)) indicate significant dynamics.
   - Complexity: \( O(n^3) \) for \( n \) points, but manageable for audio snippets with downsampling or subsampling if needed.

3. **Envelope Construction**:
   - Map persistent points’ birth values (\( b_i \)) to time indices in the signal via nearest amplitude match.
   - Build a smoothed envelope by emphasizing these points, using a windowed maximum (e.g., ±50 samples) and attack/release filtering:
     - Attack: \( \alpha_a = e^{-1 / (f_s \cdot \tau_a)} \), \( \tau_a = 0.01 \, \text{s} \).
     - Release: \( \alpha_r = e^{-1 / (f_s \cdot \tau_r)} \), \( \tau_r = 0.1 \, \text{s} \).
   - Result: An envelope that highlights topologically significant events.

4. **Gain Reduction**:
   - Convert envelope to dB: \( E_{\text{dB}}(t) = 20 \log_{10}(\max(E(t), 10^{-5})) \).
   - Apply compression:
     \[
     G_{\text{dB}}(t) = 
     \begin{cases} 
     \text{threshold} + \frac{E_{\text{dB}}(t) - \text{threshold}}{\text{ratio}} - E_{\text{dB}}(t), & \text{if } E_{\text{dB}}(t) > \text{threshold} \\
     0, & \text{otherwise}
     \end{cases}
     \]
   - Linear gain: \( G(t) = 10^{G_{\text{dB}}(t) / 20} \).

5. **Output**: Multiply the original signal by \( G(t) \) and normalize to avoid clipping.

### Implementation Details
- **Language**: Python 3.8+.
- **Dependencies**:
  - `numpy`: Array operations.
  - `scipy`: WAV I/O (`wavfile`), signal processing (`signal` for fallback).
  - `matplotlib`: Visualization.
  - `ripser`: Persistent homology computation.
  - `persim`: Wasserstein distance (unused in current version but extensible).
- **Fallback Mode**: If `ripser` fails (e.g., due to NumPy C API issues), it uses `scipy.signal.find_peaks` to detect amplitude peaks above `persistence_scale`, approximating the topological approach.

#### Algorithm Steps
1. **Input**: Read 16-bit WAV, convert stereo to mono, normalize to [-1, 1].
2. **Embedding**: Compute derivative and form 2D point cloud.
3. **Topology**:
   - With `ripser`: Run 0D homology, filter persistent features (\( p_i > \text{persistence_scale} \)).
   - Fallback: Detect peaks with height > `persistence_scale`.
4. **Envelope**: Construct and smooth envelope around significant events.
5. **Compression**: Apply gain reduction based on threshold (-20 dB) and ratio (4:1).
6. **Output**: Write compressed WAV.

#### Visualization
- **Phase Space Trajectory**: Plots amplitude vs. derivative (blue line), with persistent peaks as red dots, showing which dynamics are targeted.
- **Envelope Comparison**: Original (blue) vs. topological smoothed (red), illustrating how significant events shape the envelope.
- **Gain Reduction & Output**: Gain in dB (green) and compressed signal (blue), highlighting the sculpting effect.
- **Output File**: `topological_compressor_visualization.png`.

### Sonic Characteristics
- **Full Mode (with `ripser`)**:
  - **Sculpted Clarity**: Compresses only persistent dynamic events (e.g., loud drum hits, vocal plosives), leaving subtle contours intact.
  - **Crystalline Quality**: Peaks are chiseled down, giving a polished, transparent sound—like audio etched into a precise form.
  - **Applications**: Ideal for tracks needing detail preservation (e.g., acoustic guitar, vocals) or a refined mix polish.
- **Fallback Mode**:
  - **Approximate Sculpting**: Compresses based on amplitude peaks, less selective than homology but still effective.
  - **Smoother Dynamics**: Less “crystalline,” more conventional, but retains natural flow.

### Key Features
- **Topological Innovation**: Uses persistent homology to identify significant dynamics, a novel approach in audio compression.
- **Selective Compression**: Targets only disruptive events, preserving the signal’s essence.
- **Parameters**:
  - `threshold`: Compression threshold in dB (-20 default).
  - `ratio`: Compression strength (4.0 default).
  - `persistence_scale`: Persistence threshold (0.1 default) for feature significance.
- **Performance**: Full mode is slower due to homology computation (seconds for short clips), fallback is faster.

    
    '''