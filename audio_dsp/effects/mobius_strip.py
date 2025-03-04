import numpy as np
from scipy.io import wavfile
import librosa
import matplotlib.pyplot as plt
import librosa.display

def sonic_mobius_strip(input_signal, sample_rate=44100, fade_duration=1.0, pitch_shift=0.5, wet_mix=0.5, visualize=False):
    """
    Apply a Sonic Möbius Strip effect—signal folds forward and reverse with a twisted midpoint crossfade.
    
    Args:
        input_signal: Input audio array (mono, normalized to ±1)
        sample_rate: Sample rate in Hz (default 44100)
        fade_duration: Duration of midpoint crossfade in seconds (default 1.0s)
        pitch_shift: Pitch shift in semitones for forward/reverse (default 0.5 = ±0.5 semitones)
        wet_mix: Wet signal mix (0–1, default 0.5 = 50% effect)
        visualize: If True, plot waveform and spectrogram (default False)
    
    Returns:
        Output audio array with sonic möbius strip applied
    """
    # Ensure input is float64 and normalized
    signal = np.array(input_signal, dtype=np.float64)
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
    total_samples = len(signal)
    
    # Calculate midpoint and fade samples
    midpoint = total_samples // 2
    fade_samples = int(fade_duration * sample_rate)
    fade_start = max(0, midpoint - fade_samples // 2)
    fade_end = min(total_samples, midpoint + fade_samples // 2)
    fade_length = fade_end - fade_start
    
    # Split and process signal
    forward_part = signal[:midpoint]
    reverse_part = signal[midpoint:][::-1]  # Reverse second half
    
    # Pitch shift—forward up, reverse down
    forward_shifted = librosa.effects.pitch_shift(forward_part, sr=sample_rate, n_steps=pitch_shift)
    reverse_shifted = librosa.effects.pitch_shift(reverse_part, sr=sample_rate, n_steps=-pitch_shift)
    
    # Pad or trim to match lengths
    if len(forward_shifted) > midpoint:
        forward_shifted = forward_shifted[:midpoint]
    elif len(forward_shifted) < midpoint:
        forward_shifted = np.pad(forward_shifted, (0, midpoint - len(forward_shifted)), mode='constant')
    if len(reverse_shifted) > total_samples - midpoint:
        reverse_shifted = reverse_shifted[:total_samples - midpoint]
    elif len(reverse_shifted) < total_samples - midpoint:
        reverse_shifted = np.pad(reverse_shifted, (0, total_samples - midpoint - len(reverse_shifted)), mode='constant')
    
    # Create twisted signal
    twisted_signal = np.concatenate((forward_shifted, reverse_shifted))
    
    # Crossfade at midpoint
    fade_in = np.linspace(0, 1, fade_length)
    fade_out = np.linspace(1, 0, fade_length)
    twisted_signal[fade_start:fade_end] = signal[fade_start:fade_end] * fade_out + twisted_signal[fade_start:fade_end] * fade_in
    
    # Wet/dry mix
    output = signal * (1 - wet_mix) + twisted_signal * wet_mix
    
    # Normalize
    max_amp = np.max(np.abs(output))
    if max_amp > 0:
        output = output / max_amp
    else:
        print("Warning: Output is silent—check signal processing.")
    print(f"Output range: {np.min(output):.3f} to {np.max(output):.3f}, Wet mix: {wet_mix}, Fade duration: {fade_duration}, "
          f"Pitch shift: {pitch_shift}, Midpoint: {midpoint / sample_rate:.2f}s, Fade samples: {fade_samples}")
    
    # Visualization
    if visualize:
        times = np.linspace(0, total_samples / sample_rate, total_samples)
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(times, signal, label="Original Signal", alpha=0.5)
        plt.plot(times, output, label="Twisted Signal", color='r', alpha=0.5)
        plt.legend()
        plt.title("Waveform Before & After Sonic Möbius Strip")
        plt.subplot(2, 1, 2)
        plt.specgram(output, Fs=sample_rate, NFFT=2048, noverlap=512)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Spectrogram of Twisted Signal")
        plt.tight_layout()
        plt.show()
    
    return output

# Test it
if __name__ == "__main__":
    # Load sample data
    samplerate, data = wavfile.read("voice.wav")
    if samplerate != 44100:
        data = librosa.resample(data.astype(np.float64), orig_sr=samplerate, target_sr=44100)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data / np.max(np.abs(data))  # Normalize
    
    # Apply Sonic Möbius Strip with visualization
    effected = sonic_mobius_strip(data, sample_rate=44100, fade_duration=1.0, pitch_shift=0.5, wet_mix=0.5, visualize=True)
    
    # Save output
    wavfile.write("mobius_strip.wav", 44100, effected.astype(np.float32))
    print(f"Sonic Möbius Strip audio saved to mobius_strip.wav")