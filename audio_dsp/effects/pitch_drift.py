import numpy as np
from audio_dsp.utils import wav_io as wavfile
from scipy import interpolate
import librosa

def pitch_drift(input_signal, sample_rate=44100, drift_depth=0.1, drift_rate=0.05):
    """
    Apply a slow, random pitch drift to an input signal efficiently.
    
    Args:
        input_signal: Input audio array (mono, normalized to ±1)
        sample_rate: Sample rate in Hz (default 44100)
        drift_depth: Max pitch shift in semitones (e.g., 0.1 = ±0.1 semitones)
        drift_rate: LFO rate in Hz (e.g., 0.05 = slow drift over 20s)
    
    Returns:
        Output audio array with smooth pitch drift applied
    """
    # Ensure input is float64 and normalized
    signal = np.array(input_signal, dtype=np.float64)
    signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else signal
    
    total_samples = len(signal)
    t = np.linspace(0, total_samples / sample_rate, total_samples, endpoint=False)
    
    # Generate slow stochastic LFO for pitch shift (in semitones)
    lfo = np.random.normal(0, 1, total_samples)
    lfo = np.convolve(lfo, np.ones(500)/500, mode='same')  # Smooth over ~11ms
    pitch_shift = drift_depth * np.sin(2 * np.pi * drift_rate * t + lfo)  # ±drift_depth semitones
    
    # Convert semitones to speed (2^(semitones/12))
    speed = 2 ** (pitch_shift / 12.0)
    
    # Create warped time axis (cumulative speed change)
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
    
    # Apply pitch drift
    drifted = pitch_drift(data, sample_rate=44100, drift_depth=0.6, drift_rate=0.05)
    
    # Save output
    wavfile.write("pitch_drift.wav", 44100, drifted.astype(np.float32))
    print(f"Pitch-drifted audio saved to pitch_drift.wav")