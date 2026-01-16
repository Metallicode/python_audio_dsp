import numpy as np
from audio_dsp.utils import wav_io as wavfile

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

def melt_window_spectrum(window, slice_size=250):
    """Melt frequencies within slices for a single window."""
    fft_result = np.fft.rfft(window)
    freqs = np.fft.rfftfreq(len(window), 1 / SAMPLE_RATE)
    mag = np.abs(fft_result)
    phase = np.angle(fft_result)
    
    # Define frequency slices
    max_freq = SAMPLE_RATE / 2
    bounds = np.arange(0, max_freq + slice_size, slice_size)
    
    # Melt within each slice
    new_mag = np.zeros_like(mag)
    new_phase = np.zeros_like(phase)
    
    for i in range(len(bounds) - 1):
        low = bounds[i]
        high = bounds[i + 1]
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            avg_mag = np.mean(mag[mask]) if np.sum(mask) > 0 else 0
            avg_phase = np.mean(phase[mask]) if np.sum(mask) > 0 else 0
            # Place avg magnitude at center frequency
            center_freq_idx = np.argmin(np.abs(freqs[mask] - (low + high) / 2))
            new_mag[mask] = 0
            new_mag[np.where(mask)[0][center_freq_idx]] = avg_mag
            new_phase[mask] = avg_phase
    
    # Reconstruct window
    new_fft = new_mag * np.exp(1j * new_phase)
    return np.fft.irfft(new_fft, n=len(window))

def melt_spectrum(signal, slice_size=250, window_size=1024, hop_size=256):
    """Process signal in windows, melt spectrum, and overlap-add."""
    # Initialize output
    output = np.zeros_like(signal)
    window_count = np.zeros_like(signal)  # For normalizing overlap
    
    # Hann window for smooth overlap
    hann = np.hanning(window_size)
    
    # Process each window
    for start in range(0, len(signal) - window_size + 1, hop_size):
        window = signal[start:start + window_size] * hann
        if len(window) == window_size:  # Ensure full window
            morphed_window = melt_window_spectrum(window, slice_size=slice_size)
            output[start:start + window_size] += morphed_window
            window_count[start:start + window_size] += hann
    
    # Normalize by overlap count
    mask = window_count > 0
    output[mask] /= window_count[mask]
    
    return output

def main(input_file, output_file, slice_size=250):
    print(f"Loading {input_file}...")
    signal = load_wav(input_file)
    
    print(f"Slicing spectrum into {slice_size} Hz intervals and melting with windowing...")
    morphed_signal = melt_spectrum(signal, slice_size=slice_size)
    
    print(f"Saving to {output_file}...")
    save_wav(output_file, morphed_signal)

if __name__ == "__main__":
    input_file = "sequence.wav"  # Replace with your path
    output_file = "morphed_output.wav"
    slice_size = 50  # Hz per slice (50 sounds nice)
    
    main(input_file, output_file, slice_size)


