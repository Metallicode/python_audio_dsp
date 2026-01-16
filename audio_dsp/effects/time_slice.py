# import numpy as np
# from audio_dsp.utils import wav_io as wavfile

# SAMPLE_RATE = 44100

# def load_wav(file_path):
#     sr, data = wavfile.read(file_path)
#     if sr != SAMPLE_RATE:
#         raise ValueError(f"Sample rate must be {SAMPLE_RATE} Hz, got {sr} Hz")
#     if data.ndim > 1:
#         data = np.mean(data, axis=1)
#     return data.astype(np.float32) / 32768.0

# def save_wav(file_path, data):
#     data = np.clip(data, -1, 1)
#     wavfile.write(file_path, SAMPLE_RATE, (data * 32767).astype(np.int16))
#     print(f"Saved to {file_path}")

# def get_dominant_frequency(window):
#     """Find the dominant frequency in a time window."""
#     fft_result = np.fft.rfft(window)
#     mag = np.abs(fft_result)
#     freqs = np.fft.rfftfreq(len(window), 1 / SAMPLE_RATE)
#     dominant_idx = np.argmax(mag)  # Index of max magnitude
#     return freqs[dominant_idx]

# def time_slice_to_sines(signal, window_size=512):
#     """Slice signal in time and replace each window with its dominant sine tone."""
#     num_windows = len(signal) // window_size
#     output = np.zeros(num_windows * window_size, dtype=np.float32)
    
#     for i in range(num_windows):
#         start = i * window_size
#         end = start + window_size
#         window = signal[start:end]
        
#         if len(window) == window_size:  # Full window check
#             # Get dominant frequency
#             freq = get_dominant_frequency(window)
            
#             # Generate sine tone
#             t = np.linspace(0, window_size / SAMPLE_RATE, window_size, endpoint=False)
#             sine = np.sin(2 * np.pi * freq * t)
            
#             # Match amplitude to window's energy
#             window_energy = np.mean(np.abs(window))
#             sine = sine * window_energy
            
#             output[start:end] = sine
    
#     # Trim to match original length if needed
#     if len(output) > len(signal):
#         output = output[:len(signal)]
#     elif len(output) < len(signal):
#         output = np.pad(output, (0, len(signal) - len(output)), 'constant')
    
#     return output

# def main(input_file, output_file, window_size=512):
#     print(f"Loading {input_file}...")
#     signal = load_wav(input_file)
    
#     print(f"Slicing time into {window_size}-sample windows and chaining sine tones...")
#     morphed_signal = time_slice_to_sines(signal, window_size=window_size)
    
#     print(f"Saving to {output_file}...")
#     save_wav(output_file, morphed_signal)

# if __name__ == "__main__":
#     input_file = "sequence.wav"  # Replace with your path
#     output_file = "sine_chain_output.wav"
#     window_size = 712  # Samples per window (~11.6ms at 44.1 kHz)
    
#     main(input_file, output_file, window_size)

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

def get_dominant_frequency(window):
    """Find the dominant frequency in a time window."""
    fft_result = np.fft.rfft(window)
    mag = np.abs(fft_result)
    freqs = np.fft.rfftfreq(len(window), 1 / SAMPLE_RATE)
    dominant_idx = np.argmax(mag)
    return freqs[dominant_idx]

def time_slice_to_sines(signal, window_size=512):
    """Slice signal in time and replace each window with its dominant sine tone, no crossfade."""
    num_windows = len(signal) // window_size
    output = np.zeros(num_windows * window_size, dtype=np.float32)
    
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window = signal[start:end]
        
        if len(window) == window_size:
            # Get dominant frequency
            freq = get_dominant_frequency(window)
            
            # Generate sine tone
            t = np.linspace(0, window_size / SAMPLE_RATE, window_size, endpoint=False)
            sine = np.sin(2 * np.pi * freq * t)
            
            # Match amplitude to window's energy
            window_energy = np.mean(np.abs(window))
            sine = sine * window_energy
            
            output[start:end] = sine
    
    # Trim or pad to match original length
    if len(output) > len(signal):
        output = output[:len(signal)]
    elif len(output) < len(signal):
        output = np.pad(output, (0, len(signal) - len(output)), 'constant')
    
    return output

def main(input_file, output_file, window_size=512):
    print(f"Loading {input_file}...")
    signal = load_wav(input_file)
    
    print(f"Slicing time into {window_size}-sample windows and chaining sine tones (no crossfade)...")
    morphed_signal = time_slice_to_sines(signal, window_size=window_size)
    
    print(f"Saving to {output_file}...")
    save_wav(output_file, morphed_signal)

if __name__ == "__main__":
    input_file = "x.wav"  # Replace with your path
    output_file = "sine_chain_output.wav"
    window_size = 1024  # Samples per window 
    
    main(input_file, output_file, window_size)
