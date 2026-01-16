import numpy as np
from audio_dsp.utils import wav_io as wavfile

def filter_effect(input_signal, sample_rate=44100, cutoff=1500, resonance=0.7, q=1.0, filter_type="lowpass"):
    """
    Apply a 4-pole resonant filter effect to an input signal.
    
    Args:
        input_signal: Input audio array (mono, normalized to ±1)
        sample_rate: Sample rate in Hz (default 44100)
        cutoff: Filter cutoff frequency in Hz (50–5000)
        resonance: Resonance amplitude (0–5)
        q: Q factor (0.1–10) - bandwidth control
        filter_type: 'lowpass' or 'bandpass' (default 'lowpass')
    
    Returns:
        Output audio array with filter effect applied
    """
    # Ensure input is float64 and normalized
    signal = np.array(input_signal, dtype=np.float64)
    signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else signal
    
    total_samples = len(signal)
    nyquist = sample_rate / 2
    cutoff = np.clip(cutoff, 50, nyquist - 1)
    resonance = np.clip(resonance, 0, 5.0)
    q = np.clip(q, 0.1, 10.0)
    
    # Resonance amplitude
    res_amplitude = resonance * 0.25 * (1 + np.exp(0.5 * resonance))
    
    # Filter state
    y0, y1, y2, y3 = 0.0, 0.0, 0.0, 0.0
    output = np.zeros_like(signal)
    
    # Process signal
    for i in range(total_samples):
        wc = 2 * np.pi * cutoff / sample_rate
        alpha = np.sin(wc) / q
        input_sample = signal[i]
        
        if filter_type == "lowpass":
            feedback = res_amplitude * y3
            y0 = np.tanh(y0 + alpha * (input_sample - feedback - y0))
            y1 = np.tanh(y1 + alpha * (y0 - y1))
            y2 = np.tanh(y2 + alpha * (y1 - y2))
            y3 = np.tanh(y3 + alpha * (y2 - y3))
            y3 *= 0.98  # Slight damping
            output[i] = y3
        
        elif filter_type == "bandpass":
            feedback = res_amplitude * y3
            y0 = np.tanh(y0 + alpha * (input_sample - feedback - y0))
            y1 = np.tanh(y1 + alpha * (y0 - y1))
            y2 = np.tanh(y2 + alpha * (y1 - y2))
            y3 = np.tanh(y3 + alpha * (y2 - y3))
            y3 *= 0.98
            output[i] = y2 - y3  # Bandpass output
        
        else:
            raise ValueError("filter_type must be 'lowpass' or 'bandpass'")
    
    # Normalize
    max_amp = np.max(np.abs(output))
    if max_amp > 0:
        output = output / max_amp
    
    return output

# Test it
if __name__ == "__main__":
    import librosa
    # Load sample data
    samplerate, data = wavfile.read("input.wav")
    if samplerate != 44100:
        data = librosa.resample(data.astype(np.float64), orig_sr=samplerate, target_sr=44100)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data / np.max(np.abs(data))  # Normalize
    
    # Apply low-pass filter
    lp_filtered = filter_effect(data, sample_rate=44100, cutoff=1000, resonance=0.7, q=1.0, 
                                filter_type="lowpass")
    wavfile.write("lp_filtered.wav", 44100, lp_filtered.astype(np.float32))
    print(f"Low-pass filtered audio saved to lp_filtered.wav")
    
    # Apply bandpass filter
    bp_filtered = filter_effect(data, sample_rate=44100, cutoff=1000, resonance=1.0, q=2.0, 
                                filter_type="bandpass")
    wavfile.write("bp_filtered.wav", 44100, bp_filtered.astype(np.float32))
    print(f"Bandpass filtered audio saved to bp_filtered.wav")