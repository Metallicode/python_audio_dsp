# import numpy as np
# import wave
# import struct
# from audio_dsp.utils import wav_io as wavfile
# from scipy.signal import stft, istft
# import librosa  # For sophisticated envelope following

# def quantize_spectrum_stft(input_file, output_file, num_buckets=20, frame_length=2048, hop_length=512):
#     """
#     Read a WAV file, quantize its spectrum using STFT, and generate a new WAV file.
    
#     Parameters:
#     - input_file: Path to input WAV file
#     - output_file: Path to output WAV file
#     - num_buckets: Number of frequency buckets
#     - frame_length: STFT frame length in samples
#     - hop_length: STFT hop length in samples
#     """
#     # Read the WAV file
#     sample_rate, data = wavfile.read(input_file)
    
#     # Handle stereo by converting to mono
#     if len(data.shape) > 1:
#         data = np.mean(data, axis=1)
    
#     # Normalize input data to float between -1 and 1
#     data = data.astype(float) / np.iinfo(data.dtype).max

#     # Compute STFT
#     freqs, times, Zxx = stft(data, fs=sample_rate, nperseg=frame_length, noverlap=frame_length - hop_length)
#     magnitudes = np.abs(Zxx)
#     phases = np.angle(Zxx)

#     # Define frequency buckets
#     max_freq = sample_rate / 2
#     bucket_edges = np.linspace(0, max_freq, num_buckets + 1)
#     bucket_centers = (bucket_edges[:-1] + bucket_edges[1:]) / 2

#     # Quantize spectrum for each frame
#     quantized_Zxx = np.zeros_like(Zxx, dtype=complex)
#     for t in range(Zxx.shape[1]):  # For each time frame
#         for f in range(Zxx.shape[0]):  # For each frequency bin
#             freq = freqs[f]
#             if freq >= 0:  # Only positive frequencies
#                 magnitude = magnitudes[f, t]
#                 phase = phases[f, t]
#                 if magnitude > 0:  # Only quantize non-zero components
#                     bucket_idx = np.argmin(np.abs(bucket_centers - freq))
#                     quantized_freq = bucket_centers[bucket_idx]
#                     new_f_idx = np.argmin(np.abs(freqs - quantized_freq))
#                     if new_f_idx < Zxx.shape[0]:
#                         quantized_Zxx[new_f_idx, t] += magnitude * np.exp(1j * phase)

#     # Inverse STFT to reconstruct signal
#     _, quantized_signal = istft(quantized_Zxx, fs=sample_rate, nperseg=frame_length, noverlap=frame_length - hop_length)
#     quantized_signal = quantized_signal[:len(data)]  # Trim to original length

#     # Sophisticated envelope follower (using librosa's amplitude envelope)
#     original_envelope = librosa.effects.preemphasis(data)  # Optional pre-emphasis
#     original_envelope = np.abs(librosa.util.frame(original_envelope, frame_length=frame_length, hop_length=hop_length))
#     original_envelope = np.max(original_envelope, axis=0)  # Peak per frame
#     original_envelope = librosa.effects.deemphasis(original_envelope, coef=0.97)  # Smooth decay
    
#     # Interpolate envelope to match signal length
#     envelope_times = np.arange(0, len(data), hop_length)
#     if len(envelope_times) > len(original_envelope):
#         envelope_times = envelope_times[:len(original_envelope)]
#     elif len(envelope_times) < len(original_envelope):
#         original_envelope = original_envelope[:len(envelope_times)]
#     envelope_full = np.interp(np.arange(len(data)), envelope_times, original_envelope)

#     # Apply envelope to quantized signal
#     quantized_signal = quantized_signal * (envelope_full / np.max(np.abs(quantized_signal)))

#     # Normalize to prevent clipping
#     quantized_signal = quantized_signal / (np.max(np.abs(quantized_signal)) * 1.1)

#     # Convert to 16-bit PCM
#     quantized_signal = (quantized_signal * 32767).astype(np.int16)

#     # Write to WAV file
#     with wave.open(output_file, 'w') as wav_file:
#         wav_file.setparams((1, 2, sample_rate, len(quantized_signal), 'NONE', 'not compressed'))
#         for sample in quantized_signal:
#             wav_file.writeframes(struct.pack('h', sample))

# def main():
#     input_file = "input.wav"  # Replace with your WAV file
#     output_file = "quantized_stft.wav"
#     num_buckets = 50
#     frame_length = 2048  # ~46 ms at 44100 Hz
#     hop_length = 512     # ~11.6 ms overlap
    
#     print(f"Processing {input_file} with {num_buckets} buckets using STFT...")
#     quantize_spectrum_stft(input_file, output_file, num_buckets, frame_length, hop_length)
#     print(f"Created: {output_file}")

# if __name__ == "__main__":
#     main()



import numpy as np
import wave
import struct
from audio_dsp.utils import wav_io as wavfile
from scipy.signal import stft, istft
import librosa

def quantize_spectrum_stft_adaptive(input_file, output_file, num_buckets=20, frame_length=2048, hop_length=512):
    """
    Read a WAV file, quantize its spectrum using STFT with adaptive buckets based on average spectral energy.
    
    Parameters:
    - input_file: Path to input WAV file
    - output_file: Path to output WAV file
    - num_buckets: Number of frequency buckets
    - frame_length: STFT frame length in samples
    - hop_length: STFT hop length in samples
    """
    # Read the WAV file
    sample_rate, data = wavfile.read(input_file)
    
    # Handle stereo by converting to mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    # Normalize input data to float between -1 and 1
    data = data.astype(float) / np.iinfo(data.dtype).max

    # Compute STFT
    freqs, times, Zxx = stft(data, fs=sample_rate, nperseg=frame_length, noverlap=frame_length - hop_length)
    magnitudes = np.abs(Zxx)

    # Compute average spectrum across time
    avg_spectrum = np.mean(magnitudes, axis=1)
    pos_freqs = freqs[freqs >= 0]
    avg_spectrum = avg_spectrum[:len(pos_freqs)]

    # Normalize spectrum to treat as a probability distribution
    avg_spectrum = avg_spectrum / np.sum(avg_spectrum)

    # Cumulative energy for bucket boundaries
    cumulative_energy = np.cumsum(avg_spectrum)
    bucket_energy_step = 1.0 / num_buckets
    bucket_centers = []

    # Find bucket centers based on energy centroids
    energy_pos = 0.0
    for i in range(num_buckets):
        target_energy = (i + 1) * bucket_energy_step
        # Find frequency where cumulative energy crosses target
        idx = np.searchsorted(cumulative_energy, target_energy)
        if idx >= len(pos_freqs):
            idx = len(pos_freqs) - 1
        # Compute centroid of this energy region
        if i == 0:
            start_idx = 0
        else:
            start_idx = np.searchsorted(cumulative_energy, i * bucket_energy_step)
        energy_slice = avg_spectrum[start_idx:idx+1]
        freq_slice = pos_freqs[start_idx:idx+1]
        if np.sum(energy_slice) > 0:
            centroid = np.sum(freq_slice * energy_slice) / np.sum(energy_slice)
            bucket_centers.append(centroid)
        else:
            bucket_centers.append(pos_freqs[idx])  # Fallback to boundary

    bucket_centers = np.array(bucket_centers)
    print(f"Adaptive bucket centers: {bucket_centers}")

    # Quantize spectrum for each frame
    quantized_Zxx = np.zeros_like(Zxx, dtype=complex)
    for t in range(Zxx.shape[1]):  # For each time frame
        for f in range(Zxx.shape[0]):  # For each frequency bin
            freq = freqs[f]
            if freq >= 0:  # Only positive frequencies
                magnitude = magnitudes[f, t]
                phase = np.angle(Zxx[f, t])
                if magnitude > 0:
                    bucket_idx = np.argmin(np.abs(bucket_centers - freq))
                    quantized_freq = bucket_centers[bucket_idx]
                    new_f_idx = np.argmin(np.abs(freqs - quantized_freq))
                    if new_f_idx < Zxx.shape[0]:
                        quantized_Zxx[new_f_idx, t] += magnitude * np.exp(1j * phase)

    # Inverse STFT
    _, quantized_signal = istft(quantized_Zxx, fs=sample_rate, nperseg=frame_length, noverlap=frame_length - hop_length)
    quantized_signal = quantized_signal[:len(data)]  # Trim to original length

    # Sophisticated envelope follower
    original_envelope = librosa.effects.preemphasis(data)
    original_envelope = np.abs(librosa.util.frame(original_envelope, frame_length=frame_length, hop_length=hop_length))
    original_envelope = np.max(original_envelope, axis=0)
    original_envelope = librosa.effects.deemphasis(original_envelope, coef=0.97)
    
    # Interpolate envelope
    envelope_times = np.arange(0, len(data), hop_length)
    if len(envelope_times) > len(original_envelope):
        envelope_times = envelope_times[:len(original_envelope)]
    elif len(envelope_times) < len(original_envelope):
        original_envelope = original_envelope[:len(envelope_times)]
    envelope_full = np.interp(np.arange(len(data)), envelope_times, original_envelope)

    # Apply envelope
    quantized_signal = quantized_signal * (envelope_full / np.max(np.abs(quantized_signal)))

    # Normalize
    quantized_signal = quantized_signal / (np.max(np.abs(quantized_signal)) * 1.1)

    # Convert to 16-bit PCM
    quantized_signal = (quantized_signal * 32767).astype(np.int16)

    # Write to WAV file
    with wave.open(output_file, 'w') as wav_file:
        wav_file.setparams((1, 2, sample_rate, len(quantized_signal), 'NONE', 'not compressed'))
        for sample in quantized_signal:
            wav_file.writeframes(struct.pack('h', sample))

def main():
    input_file = "input.wav"  # Replace with your WAV file
    output_file = "quantized_adaptive.wav"
    num_buckets = 50
    frame_length = 2048
    hop_length = 512
    
    print(f"Processing {input_file} with {num_buckets} adaptive buckets using STFT...")
    quantize_spectrum_stft_adaptive(input_file, output_file, num_buckets, frame_length, hop_length)
    print(f"Created: {output_file}")

if __name__ == "__main__":
    main()
