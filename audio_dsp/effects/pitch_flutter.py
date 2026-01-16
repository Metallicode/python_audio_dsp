import numpy as np
from audio_dsp.utils import wav_io as wavfile
from scipy import interpolate
import librosa

def flutter_effect(input_signal, sample_rate=44100, base_rate=5.0, rate_diff=1.0, depth=0.05):
    """
    Apply a flutter effect with dual-sine LFO modulation to an input signal.
    
    Args:
        input_signal: Input audio array (mono, normalized to ±1)
        sample_rate: Sample rate in Hz (default 44100)
        base_rate: Base LFO frequency in Hz (e.g., 5.0 = 5 Hz)
        rate_diff: Frequency difference between LFOs in Hz (e.g., 1.0 = 1 Hz diff)
        depth: Modulation depth (e.g., 0.05 = ±5% speed variation, ~1-10 Hz range)
    
    Returns:
        Output audio array with flutter effect applied
    """
    # Ensure input is float64 and normalized
    signal = np.array(input_signal, dtype=np.float64)
    signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else signal
    
    total_samples = len(signal)
    t = np.linspace(0, total_samples / sample_rate, total_samples, endpoint=False)
    
    # Generate dual-sine LFO for flutter
    lfo1 = np.sin(2 * np.pi * base_rate * t)  # Base frequency (e.g., 5 Hz)
    lfo2 = np.sin(2 * np.pi * (base_rate + rate_diff) * t)  # Offset frequency (e.g., 6 Hz)
    lfo = (lfo1 + lfo2) / 2.0  # Combine, average amplitude
    
    # Map LFO to speed variation (±depth, e.g., 0.95 to 1.05 for depth=0.05)
    speed = 1.0 + (depth * lfo)  # Speed range: 1 ± depth
    
    # Create warped time axis
    warped_time = np.cumsum(1.0 / speed)  # Inverse speed = time stretch
    warped_time = warped_time * (total_samples / warped_time[-1])  # Normalize to original length
    
    # Interpolate original signal onto warped time
    interp_func = interpolate.interp1d(np.arange(total_samples), signal, kind='linear', 
                                       bounds_error=False, fill_value=0)
    output = interp_func(warped_time)
    
    # Normalize
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
    
    # Apply flutter effect
    fluttered = flutter_effect(data, sample_rate=44100, base_rate=5.0, rate_diff=1.0, depth=0.05)
    
    # Save output
    wavfile.write("flutter_effect.wav", 44100, fluttered.astype(np.float32))
    print(f"Flutter-effect audio saved to flutter_effect.wav")