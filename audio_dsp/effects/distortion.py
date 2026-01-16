import numpy as np
from audio_dsp.utils import wav_io as wavfile
from scipy.signal import stft, istft

SAMPLE_RATE = 44100

def load_wav(file_path):
    sr, data = wavfile.read(file_path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Sample rate must be {SAMPLE_RATE} Hz, got {sr} Hz")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32) / 32768.0

def save_wav(file_path, data):
    data = np.clip(data, -1, 1)
    wavfile.write(file_path, SAMPLE_RATE, (data * 32767).astype(np.int16))
    print(f"Saved to {file_path}")

# Existing distortions
def fuzz_distortion(signal, gain=10.0, threshold=0.3, mix=1.0):
    distorted = signal * gain
    distorted = np.where(distorted > threshold, threshold, distorted)
    distorted = np.where(distorted < -threshold, -threshold, distorted)
    distorted = np.sign(distorted) * (1 - np.exp(-np.abs(distorted)))
    return mix * distorted + (1 - mix) * signal

def overdrive_distortion(signal, gain=2.0, threshold=0.5, mix=1.0):
    distorted = signal * gain
    distorted = np.tanh(distorted / threshold) * threshold
    return mix * distorted + (1 - mix) * signal

def saturation_distortion(signal, gain=1.5, threshold=0.7, mix=1.0):
    distorted = signal * gain
    distorted = np.arctan(distorted / threshold) * (2 / np.pi) * threshold
    return mix * distorted + (1 - mix) * signal

# New distortions
def cubic_distortion(signal, gain=1.0, k=0.1, mix=1.0):
    distorted = signal * gain
    distorted = distorted + k * distorted**3
    distorted = np.clip(distorted, -1, 1)
    return mix * distorted + (1 - mix) * signal

def hard_clip_distortion(signal, gain=5.0, threshold=0.5, mix=1.0):
    distorted = signal * gain
    distorted = np.clip(distorted, -threshold, threshold)
    return mix * distorted + (1 - mix) * signal

def wavefold_distortion(signal, gain=5.0, mix=1.0):
    distorted = signal * gain
    distorted = np.sin(distorted * np.pi)
    return mix * distorted + (1 - mix) * signal

def bitcrush_distortion(signal, gain=1.0, bits=8, mix=1.0):
    step_size = 1.0 / (2 ** (bits - 1))
    distorted = signal * gain
    distorted = np.round(distorted / step_size) * step_size
    return mix * distorted + (1 - mix) * signal

def asymmetric_distortion(signal, gain=3.0, threshold_pos=0.6, threshold_neg=0.4, mix=1.0):
    distorted = signal * gain
    distorted = np.where(distorted > threshold_pos, threshold_pos, distorted)
    distorted = np.where(distorted < -threshold_neg, -threshold_neg, distorted)
    return mix * distorted + (1 - mix) * signal

def logistic_distortion(signal, gain=5.0, mix=1.0):
    distorted = 2 / (1 + np.exp(-gain * signal)) - 1
    return mix * distorted + (1 - mix) * signal

def poly_distortion(signal, gain=2.0, k1=0.2, k2=0.1, mix=1.0):
    distorted = signal * gain
    distorted = distorted + k1 * distorted**3 + k2 * distorted**5
    distorted = np.clip(distorted, -1, 1)
    return mix * distorted + (1 - mix) * signal

def triangle_fold_distortion(signal, gain=5.0, mix=1.0):
    distorted = signal * gain
    distorted = 2 * np.abs(distorted % 2 - 1) - 1
    return mix * distorted + (1 - mix) * signal

def sawtooth_fold_distortion(signal, gain=5.0, mix=1.0):
    distorted = signal * gain
    distorted = (distorted % 1) * 2 - 1
    return mix * distorted + (1 - mix) * signal

def chebyshev_fold_distortion(signal, gain=2.0, n=2, mix=1.0):
    distorted = signal * gain
    distorted = np.clip(distorted, -1, 1)
    distorted = np.cos(n * np.arccos(distorted))
    return mix * distorted + (1 - mix) * signal

def parabolic_fold_distortion(signal, gain=5.0, mix=1.0):
    distorted = signal * gain
    distorted = 1 - 2 * ((distorted % 1) ** 2)
    return mix * distorted + (1 - mix) * signal

def exp_fold_distortion(signal, gain=5.0, mix=1.0):
    distorted = signal * gain
    distorted = np.sign(distorted) * (1 - np.exp(-np.abs(distorted))) % 1 * 2 - 1
    return mix * distorted + (1 - mix) * signal

def fractal_fold_distortion(signal, gain=5.0, mix=1.0):
    distorted = signal * gain
    distorted = np.sin(distorted + np.sin(distorted))
    return mix * distorted + (1 - mix) * signal

def mirror_fold_distortion(signal, gain=5.0, mix=1.0):
    distorted = signal * gain
    distorted = np.abs(np.sin(distorted)) * np.sign(distorted)
    return mix * distorted + (1 - mix) * signal

def dynamic_triangle_fold_distortion(signal, base_gain=5.0, sensitivity=2.0, mix=1.0):
    """Triangle fold where folding intensity scales with input amplitude."""
    # Dynamic gain based on instantaneous amplitude
    dynamic_gain = base_gain * (1 + sensitivity * np.abs(signal))
    distorted = signal * dynamic_gain
    distorted = 2 * np.abs(distorted % 2 - 1) - 1
    return mix * distorted + (1 - mix) * signal

def frequency_lock_distortion(signal, gain=10.0, num_freqs=3, bits=8, mix=1.0):
    """Distortion that locks onto dominant frequencies, amplifies them brutally, and quantizes."""
    # Adjust nperseg dynamically based on signal length
    signal_len = len(signal)
    nperseg = min(1024, signal_len)  # Use smaller of 1024 or signal length
    noverlap = nperseg // 2
    
    # Compute STFT
    f, t, Zxx = stft(signal, fs=SAMPLE_RATE, nperseg=nperseg, noverlap=noverlap)
    
    # Magnitude spectrum
    mag = np.abs(Zxx)
    
    # For each time frame, lock onto top frequencies
    distorted_Zxx = np.zeros_like(Zxx, dtype=complex)
    for i in range(mag.shape[1]):
        freq_indices = np.argpartition(mag[:, i], -num_freqs)[-num_freqs:]
        octave_indices = []
        for idx in freq_indices:
            octave_indices.extend([idx, min(idx * 2, mag.shape[0] - 1)])  # Fundamental + octave
        octave_indices = np.unique(octave_indices)
        
        for idx in octave_indices:
            distorted_Zxx[idx, i] = Zxx[idx, i] * gain
            mag_val = np.abs(distorted_Zxx[idx, i])
            if mag_val > 1.0:
                distorted_Zxx[idx, i] *= 1.0 / mag_val
    
    # Quantize the magnitude
    step_size = 1.0 / (2 ** (bits - 1))
    mag_distorted = np.abs(distorted_Zxx)
    mag_distorted = np.round(mag_distorted / step_size) * step_size
    phase = np.angle(distorted_Zxx)
    distorted_Zxx = mag_distorted * np.exp(1j * phase)
    
    # Reconstruct signal
    _, distorted_signal = istft(distorted_Zxx, fs=SAMPLE_RATE, nperseg=nperseg, noverlap=noverlap)
    
    # Ensure output matches input length
    if len(distorted_signal) > signal_len:
        distorted_signal = distorted_signal[:signal_len]
    elif len(distorted_signal) < signal_len:
        distorted_signal = np.pad(distorted_signal, (0, signal_len - len(distorted_signal)), 'constant')
    
    return mix * distorted_signal + (1 - mix) * signal

def generate_transfer_function(distortion_func, *args, input_range=(-1, 1), points=1000):
    x = np.linspace(input_range[0], input_range[1], points)
    y = distortion_func(x, *args)
    return x, y

def plot_effects(input_signal, output_signal, distortion_name, transfer_x, transfer_y, *args):
    import matplotlib.pyplot as plt  # Lazy import for optional dependency
    t = np.linspace(0, 1, SAMPLE_RATE)
    sine = np.sin(2 * np.pi * 1 * t)
    distorted_sine = globals()[f"{distortion_name.lower()}_distortion"](sine, *args)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(transfer_x, transfer_y, label=f"{distortion_name} Transfer", color='red')
    ax1.plot(transfer_x, transfer_x, label="Linear (No Effect)", color='blue', linestyle='--')
    ax1.set_title(f"{distortion_name} Transfer Function")
    ax1.set_xlabel("Input")
    ax1.set_ylabel("Output")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(t[:int(SAMPLE_RATE * 0.1)], sine[:int(SAMPLE_RATE * 0.1)], label="Pure Sine", color='blue')
    ax2.plot(t[:int(SAMPLE_RATE * 0.1)], distorted_sine[:int(SAMPLE_RATE * 0.1)], label=f"{distortion_name} Sine", color='red')
    ax2.set_title(f"{distortion_name} Effect on Sine Wave")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def process_audio(input_file, output_file, distortion_type="overdrive", *args):
    print(f"Loading {input_file}...")
    signal = load_wav(input_file)

    distortion_funcs = {
        "fuzz": fuzz_distortion,
        "overdrive": overdrive_distortion,
        "saturation": saturation_distortion,
        "cubic": cubic_distortion,
        "hard_clip": hard_clip_distortion,
        "wavefold": wavefold_distortion,
        "bitcrush": bitcrush_distortion,
        "asymmetric": asymmetric_distortion,
        "logistic": logistic_distortion,
        "poly": poly_distortion,
        "triangle_fold": triangle_fold_distortion,
        "sawtooth_fold": sawtooth_fold_distortion,
        "chebyshev_fold": chebyshev_fold_distortion,
        "parabolic_fold": parabolic_fold_distortion,
        "exp_fold": exp_fold_distortion,
        "fractal_fold": fractal_fold_distortion,
        "mirror_fold": mirror_fold_distortion,
        "dynamic_triangle_fold": dynamic_triangle_fold_distortion,
        "frequency_lock": frequency_lock_distortion
    }
    if distortion_type.lower() not in distortion_funcs:
        raise ValueError(f"Distortion type must be one of {list(distortion_funcs.keys())}")
    distortion_func = distortion_funcs[distortion_type.lower()]

    print(f"Applying {distortion_type} distortion...")
    distorted_signal = distortion_func(signal, *args)

    save_wav(output_file, distorted_signal)

    transfer_x, transfer_y = generate_transfer_function(distortion_func, *args)
    plot_effects(signal, distorted_signal, distortion_type.capitalize(), transfer_x, transfer_y, *args)

def main():
    input_file = "sine_chain_output.wav"
    distortions = [
        ("fuzz", 10.0, 0.3, 1.0),
        ("overdrive", 2.0, 0.5, 1.0),
        ("saturation", 1.5, 0.7, 1.0),
        ("cubic", 1.0, 0.1, 1.0),
        ("hard_clip", 5.0, 0.5, 1.0),
        ("wavefold", 5.0, 1.0),
        ("bitcrush", 1.0, 8, 1.0),
        ("asymmetric", 3.0, 0.6, 0.4, 1.0),
        ("logistic", 5.0, 1.0),
        ("poly", 2.0, 0.2, 0.1, 1.0),
        ("triangle_fold", 5.0, 1.0),
        ("sawtooth_fold", 5.0, 1.0),
        ("chebyshev_fold", 2.0, 3, 1.0),  # n=3 for third-order
        ("parabolic_fold", 5.0, 1.0),
        ("exp_fold", 5.0, 1.0),
        ("fractal_fold", 5.0, 1.0),
        ("mirror_fold", 5.0, 1.0),
        ("dynamic_triangle_fold", 5.0, 2.0, 1.0),
        ("frequency_lock", 10.0, 3, 8, 1.0)
    ]
    
    for distortion_type, *params in distortions:
        output_file = f"output_{distortion_type}.wav"
        process_audio(input_file, output_file, distortion_type, *params)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
