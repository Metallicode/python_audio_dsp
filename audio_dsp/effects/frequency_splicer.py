import numpy as np
from audio_dsp.utils import wav_io as wavfile
import matplotlib.pyplot as plt

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

def splice_spectrum(signal, interval_type='octave', base_freq=100):
    """Splice and average the frequency spectrum."""
    # FFT
    fft_result = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1 / SAMPLE_RATE)
    mag = np.abs(fft_result)
    phase = np.angle(fft_result)
    
    # Define frequency intervals
    if interval_type == 'octave':
        # Octave intervals starting from base_freq (e.g., 100–200, 200–400, 400–800 Hz)
        max_freq = SAMPLE_RATE / 2  # Nyquist
        bounds = [base_freq]
        while bounds[-1] < max_freq:
            bounds.append(bounds[-1] * 2)
        bounds.append(max_freq)
    elif interval_type == 'fixed':
        # Fixed intervals (e.g., 500 Hz wide)
        bounds = np.arange(0, SAMPLE_RATE / 2 + 500, 500)
    
    # Splice and average
    new_mag = np.zeros_like(mag)
    new_phase = np.zeros_like(phase)
    
    for i in range(len(bounds) - 1):
        low = bounds[i]
        high = bounds[i + 1]
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            avg_mag = np.mean(mag[mask]) if np.sum(mask) > 0 else 0
            avg_phase = np.mean(phase[mask]) if np.sum(mask) > 0 else 0
            new_mag[mask] = avg_mag
            new_phase[mask] = avg_phase
    
    # Reconstruct complex spectrum
    new_fft = new_mag * np.exp(1j * new_phase)
    morphed_signal = np.fft.irfft(new_fft, n=len(signal))
    
    return morphed_signal

def plot_signals(original, generated):
    plt.figure(figsize=(12, 6))
    t_orig = np.linspace(0, len(original) / SAMPLE_RATE, len(original))
    t_gen = np.linspace(0, len(original) / SAMPLE_RATE, len(generated))
    plt.plot(t_orig, original, label="Original", color='blue', alpha=0.5)
    plt.plot(t_gen, generated, label="Generated", color='red')
    plt.title("Original vs. Frequency-Spliced Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

def create_morphed_signal(input_file, output_file, interval_type='octave', base_freq=100):
    print(f"Loading {input_file}...")
    signal = load_wav(input_file)
    signal_length = len(signal)
    
    print("Splicing and averaging spectrum...")
    morphed_signal = splice_spectrum(signal, interval_type=interval_type, base_freq=base_freq)
    
    # Match original length (should be close already, but just in case)
    if len(morphed_signal) > signal_length:
        morphed_signal = morphed_signal[:signal_length]
    elif len(morphed_signal) < signal_length:
        morphed_signal = np.pad(morphed_signal, (0, signal_length - len(morphed_signal)), 'constant')
    
    save_wav(output_file, morphed_signal)
    plot_signals(signal, morphed_signal)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python frequency_splicer.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "morphed_output.wav"
    create_morphed_signal(input_file, output_file, interval_type='octave', base_freq=100)