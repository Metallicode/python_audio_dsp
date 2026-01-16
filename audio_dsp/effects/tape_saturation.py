import numpy as np
from audio_dsp.utils import wav_io as wavfile
import librosa

def tape_saturation(input_signal, sample_rate=44100, drive=2.0, warmth=0.1, output_level=1.0):
    """
    Apply a tape saturation effect to an input signal.
    
    Args:
        input_signal: Input audio array (mono, normalized to ±1)
        sample_rate: Sample rate in Hz (default 44100)
        drive: Input gain factor (e.g., 2.0 = 2x gain, more saturation)
        warmth: Harmonic distortion amount (e.g., 0.1 = subtle, 0.5 = strong)
        output_level: Output gain factor (e.g., 1.0 = full, 0.5 = half)
    
    Returns:
        Output audio array with tape saturation applied
    """
    # Ensure input is float64 and normalized
    signal = np.array(input_signal, dtype=np.float64)
    signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else signal
    
    # Apply drive (pre-gain)
    driven_signal = signal * drive
    
    # Tape saturation: tanh soft clip + cubic warmth
    saturated = np.tanh(driven_signal) + warmth * (driven_signal ** 3)
    
    # Blend dry and wet (50% each) for natural feel
    output = 0.5 * signal + 0.5 * saturated
    
    # Apply output level and normalize
    output = output * output_level
    max_amp = np.max(np.abs(output))
    if max_amp > 0:
        output = output / max_amp
    
    return output

# Test it
if __name__ == "__main__":
    # Load sample data
    samplerate, data = wavfile.read("input.wav")
    if samplerate != 44100:
        data = librosa.resample(data.astype(np.float64), orig_sr=samplerate, target_sr=44100)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data / np.max(np.abs(data))  # Normalize
    
    # Apply tape saturation
    saturated = tape_saturation(data, sample_rate=44100, drive=3.0, warmth=0.4, output_level=1.0)
    
    # Save output
    wavfile.write("tape_saturation.wav", 44100, saturated.astype(np.float32))
    print(f"Tape-saturated audio saved to tape_saturation.wav")