# import numpy as np
# import matplotlib.pyplot as plt
# import librosa.display
# from audio_dsp.utils import wav_io as wavfile

# def quantized_lfo_arpeggio(sample_rate=44100, freq_set=None, 
#                            lfo_rate=5.0, glide_rate=0.1, output_length=10.0, 
#                            waveform='sine', visualize=False):
#     """
#     Generate a quantized LFO-based arpeggio effect.
    
#     Args:
#         sample_rate: Sample rate in Hz
#         freq_set: List of predefined frequencies in Hz
#         lfo_rate: LFO speed in Hz
#         glide_rate: Glide time between frequencies in seconds
#         output_length: Output duration in seconds
#         waveform: LFO waveform ('random', 'sine', 'saw', 'triangle')
#         visualize: If True, plot waveform and spectrogram
    
#     Returns:
#         Output audio array
#     """
#     # Sort frequencies and get min/max for full range mapping
#     if freq_set is None:
#         freq_set = np.array([110, 220, 330, 440, 550, 660, 770, 880])
#     freq_set = np.sort(freq_set)
#     min_freq, max_freq = freq_set[0], freq_set[-1]
#     num_steps = len(freq_set)

#     # Time axis
#     total_samples = int(output_length * sample_rate)
#     t = np.linspace(0, output_length, total_samples, endpoint=False)

#     # Generate LFO waveform (normalized to -1 to 1)
#     if waveform == 'random':
#         lfo_raw = np.random.uniform(-1, 1, total_samples)
#     elif waveform == 'sine':
#         lfo_raw = np.sin(2 * np.pi * lfo_rate * t)
#     elif waveform == 'saw':
#         lfo_raw = 2 * (t * lfo_rate % 1) - 1
#     elif waveform == 'triangle':
#         lfo_raw = 2 * np.abs(2 * (t * lfo_rate % 1) - 1) - 1
#     else:
#         raise ValueError("Waveform must be 'random', 'sine', 'saw', or 'triangle'")

#     print(f"LFO raw range: {np.min(lfo_raw):.2f} to {np.max(lfo_raw):.2f}")

#     # **Ensure LFO is mapped to full frequency range**
#     lfo_scaled = np.interp(lfo_raw, [-1, 1], [min_freq, max_freq])
#     print(f"LFO mapped to frequency range: {np.min(lfo_scaled):.2f} to {np.max(lfo_scaled):.2f}")

#     # **Quantize the LFO to the nearest available frequency**
#     quantized_freqs = np.array([freq_set[np.argmin(np.abs(freq_set - f))] for f in lfo_scaled])
#     print(f"Quantized frequency range: {np.min(quantized_freqs):.2f} to {np.max(quantized_freqs):.2f}")

#     # Apply portamento glide dynamically
#     glide_samples = int(glide_rate * sample_rate)
#     freq_output = np.copy(quantized_freqs)

#     for i in range(1, total_samples):
#         prev_freq = freq_output[i - 1]
#         target_freq = quantized_freqs[i]
        
#         # Glide transition: Linear interpolation
#         if i % glide_samples < glide_samples:
#             alpha = (i % glide_samples) / glide_samples
#             freq_output[i] = prev_freq + (target_freq - prev_freq) * alpha
#         else:
#             freq_output[i] = target_freq

#     print(f"Final Frequency Output Range: {np.min(freq_output):.2f} to {np.max(freq_output):.2f}")

#     # Generate output sine wave
#     output = np.sin(2 * np.pi * freq_output * t)

#     # Normalize
#     output = output / np.max(np.abs(output)) if np.max(np.abs(output)) > 0 else output
#     print(f"Output range: {np.min(output):.3f} to {np.max(output):.3f}, Output length: {output_length}s")
    
#     # Visualization
#     if visualize:
#         plt.figure(figsize=(10, 5))
        
#         # Waveform plot
#         plt.subplot(2, 1, 1)
#         plt.plot(t, freq_output, label="Glided Frequency", color='r', alpha=0.5)
#         plt.scatter(t[::1000], quantized_freqs[::1000], color='g', label="Quantized Steps", alpha=0.3)
#         plt.legend()
#         plt.title(f"Waveform and Frequency of Quantized LFO ({waveform})")

#         # Spectrogram
#         plt.subplot(2, 1, 2)
#         plt.specgram(output, Fs=sample_rate, NFFT=2048, noverlap=512)
#         plt.xlabel("Time (s)")
#         plt.ylabel("Frequency (Hz)")
#         plt.title("Spectrogram of Quantized LFO Signal")
        
#         plt.tight_layout()
#         plt.show()
    
#     return output

# # Test it
# if __name__ == "__main__":
#     sample_rate = 44100
#     freq_set = [x for x in range(50, 1000, 50)]  # Custom scale
#     lfo_rate = 0.1  # Hz (5 note changes per second)
#     glide_rate = 0.1  # 100ms portamento
#     output_length = 10.0  # 10s output
#     waveform = 'random'  # Test sine mode
    
#     output = quantized_lfo_arpeggio(sample_rate=sample_rate, freq_set=freq_set, 
#                                     lfo_rate=lfo_rate, glide_rate=glide_rate, 
#                                     output_length=output_length, waveform=waveform, visualize=True)
    
#     wavfile.write("quantized_lfo_arpeggio.wav", sample_rate, output.astype(np.float32))
#     print(f"Quantized LFO Arpeggio audio saved to quantized_lfo_arpeggio.wav")


import numpy as np
from audio_dsp.utils import wav_io as wavfile

def quantized_lfo_arpeggio(sample_rate=44100, freq_set=None, 
                           lfo_rate=5.0, glide_rate=0.1, output_length=10.0, 
                           waveform='sine', visualize=False):
    """
    Generate a quantized LFO-based arpeggio effect with smooth stochastic LFO.
    
    Args:
        sample_rate: Sample rate in Hz
        freq_set: List of predefined frequencies in Hz
        lfo_rate: LFO speed in Hz
        glide_rate: Glide time between frequencies in seconds
        output_length: Output duration in seconds
        waveform: LFO waveform ('random_walk', 'sine', 'saw', 'triangle')
        visualize: If True, plot waveform and spectrogram
    
    Returns:
        Output audio array
    """
    # Sort frequencies and get min/max for full range mapping
    if freq_set is None:
        freq_set = np.array([110, 220, 330, 440, 550, 660, 770, 880])
    freq_set = np.sort(freq_set)
    min_freq, max_freq = freq_set[0], freq_set[-1]
    num_steps = len(freq_set)

    # Time axis
    total_samples = int(output_length * sample_rate)
    t = np.linspace(0, output_length, total_samples, endpoint=False)

    # Generate LFO waveform (normalized to -1 to 1)
    if waveform == 'random_walk':
        # **Generate a slow-moving stochastic LFO instead of noise**
        steps = int(output_length * lfo_rate)  # Number of LFO steps
        lfo_points = np.cumsum(np.random.uniform(-0.1, 0.1, steps))  # Slow random walk
        lfo_points = (lfo_points - np.min(lfo_points)) / (np.max(lfo_points) - np.min(lfo_points))  # Normalize to 0-1
        lfo_points = (lfo_points * 2) - 1  # Scale to -1 to 1
        lfo_raw = np.interp(t, np.linspace(0, output_length, steps), lfo_points)  # Interpolate for smooth LFO
    elif waveform == 'sine':
        lfo_raw = np.sin(2 * np.pi * lfo_rate * t)
    elif waveform == 'saw':
        lfo_raw = 2 * (t * lfo_rate % 1) - 1
    elif waveform == 'triangle':
        lfo_raw = 2 * np.abs(2 * (t * lfo_rate % 1) - 1) - 1
    else:
        raise ValueError("Waveform must be 'random_walk', 'sine', 'saw', or 'triangle'")

    print(f"LFO raw range: {np.min(lfo_raw):.2f} to {np.max(lfo_raw):.2f}")

    # **Ensure LFO is mapped to full frequency range**
    lfo_scaled = np.interp(lfo_raw, [-1, 1], [min_freq, max_freq])
    print(f"LFO mapped to frequency range: {np.min(lfo_scaled):.2f} to {np.max(lfo_scaled):.2f}")

    # **Quantize the LFO to the nearest available frequency**
    quantized_freqs = np.array([freq_set[np.argmin(np.abs(freq_set - f))] for f in lfo_scaled])
    print(f"Quantized frequency range: {np.min(quantized_freqs):.2f} to {np.max(quantized_freqs):.2f}")

    # **Ensure smooth, click-free gliding using sine phase interpolation**
    glide_samples = int(glide_rate * sample_rate)
    freq_output = np.copy(quantized_freqs)
    phase = np.zeros(total_samples)

    for i in range(1, total_samples):
        prev_freq = freq_output[i - 1]
        target_freq = quantized_freqs[i]

        # Smooth sine-based glide between notes
        if i % glide_samples < glide_samples:
            alpha = (1 - np.cos((i % glide_samples) / glide_samples * np.pi)) / 2  # Smooth cosine ease-in-out
            freq_output[i] = prev_freq + (target_freq - prev_freq) * alpha
        else:
            freq_output[i] = target_freq

        # Update phase for continuous waveform
        phase[i] = phase[i - 1] + 2 * np.pi * freq_output[i] / sample_rate

    print(f"Final Frequency Output Range: {np.min(freq_output):.2f} to {np.max(freq_output):.2f}")

    # **Generate output sine wave without clicks**
    output = np.sin(phase)

    # Normalize
    output = output / np.max(np.abs(output)) if np.max(np.abs(output)) > 0 else output
    print(f"Output range: {np.min(output):.3f} to {np.max(output):.3f}, Output length: {output_length}s")
    
    # Visualization
    if visualize:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        
        # Waveform plot
        plt.subplot(2, 1, 1)
        plt.plot(t, freq_output, label="Glided Frequency", color='r', alpha=0.5)
        plt.scatter(t[::1000], quantized_freqs[::1000], color='g', label="Quantized Steps", alpha=0.3)
        plt.legend()
        plt.title(f"Waveform and Frequency of Quantized LFO ({waveform})")

        # Spectrogram
        plt.subplot(2, 1, 2)
        plt.specgram(output, Fs=sample_rate, NFFT=2048, noverlap=512)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Spectrogram of Quantized LFO Signal")
        
        plt.tight_layout()
        plt.show()
    
    return output

# Test it
if __name__ == "__main__":
    sample_rate = 44100
    freq_set = [x for x in range(50, 1000, 50)]  # Custom scale
    lfo_rate = 3.0  # Hz (2 note changes per second)
    glide_rate = 0.02  # 200ms portamento
    output_length = 10.0  # 10s output
    waveform = 'random_walk'  # Use new stochastic LFO mode
    
    output = quantized_lfo_arpeggio(sample_rate=sample_rate, freq_set=freq_set, 
                                    lfo_rate=lfo_rate, glide_rate=glide_rate, 
                                    output_length=output_length, waveform=waveform, visualize=True)
    
    wavfile.write("quantized_lfo_arpeggio.wav", sample_rate, output.astype(np.float32))
    print(f"Quantized LFO Arpeggio audio saved to quantized_lfo_arpeggio.wav")
