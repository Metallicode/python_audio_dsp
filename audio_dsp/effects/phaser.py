import numpy as np
from audio_dsp.utils import wav_io as wavfile
import librosa

def phaser_flanger_effect(input_signal, sample_rate=44100, phaser_rate=0.5, phaser_depth=0.8, 
                          phaser_stages=4, flanger_delay_base=0.005, flanger_delay_depth=0.005, 
                          flanger_rate=0.25, phaser_mix=0.7, flanger_mix=0.7, wet_mix=0.7, wet_gain=1.5):
    """
    Apply a strong combined phaser and flanger effect to an input signal.
    
    Args:
        input_signal: Input audio array (mono, normalized to ±1)
        sample_rate: Sample rate in Hz (default 44100)
        phaser_rate: Phaser LFO rate in Hz (e.g., 0.5 = slow sweep)
        phaser_depth: Phaser notch depth (0–1, e.g., 0.8 = deep)
        phaser_stages: Number of all-pass stages (e.g., 4 = rich phasing)
        flanger_delay_base: Base delay time in seconds (e.g., 0.005 = 5ms)
        flanger_delay_depth: Delay modulation depth in seconds (e.g., 0.005 = ±5ms)
        flanger_rate: Flanger LFO rate in Hz (e.g., 0.25 = slower swoosh)
        phaser_mix: Balance between phaser and dry (0–1, default 0.7 = mostly phaser)
        flanger_mix: Balance between flanger and dry (0–1, default 0.7 = mostly flanger)
        wet_mix: Overall wet/dry mix (0–1, default 0.7 = wetter)
        wet_gain: Wet signal gain factor (e.g., 1.5 = boosted wet signal)
    
    Returns:
        Output audio array with phaser/flanger effect applied
    """
    # Ensure input is float64 and normalized
    signal = np.array(input_signal, dtype=np.float64)
    signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else signal
    total_samples = len(signal)
    t = np.linspace(0, total_samples / sample_rate, total_samples, endpoint=False)
    
    # Phaser: All-pass filters with LFO
    phaser_lfo = np.sin(2 * np.pi * phaser_rate * t)  # LFO for phaser cutoff
    phaser_out = signal.copy()
    for _ in range(phaser_stages):
        delayed = np.zeros_like(phaser_out)
        delayed[0] = phaser_out[0]  # Initialize
        delayed[1] = phaser_out[1]  # Avoid out-of-bounds
        for i in range(2, total_samples):
            # Compute coefficients per sample
            cutoff = 500 + 4500 * (1 + phaser_depth * phaser_lfo[i])  # Sweep 500–5000 Hz
            omega = 2 * np.pi * cutoff / sample_rate
            alpha = np.sin(omega) / (np.cos(omega) + 1.5)  # Stronger phase shift
            b0, b1, b2 = alpha, -1, 0
            a0, a1, a2 = 1, -alpha, 0
            delayed[i] = (b0 * phaser_out[i] + b1 * phaser_out[i-1] + b2 * phaser_out[i-2] - 
                          a1 * delayed[i-1] - a2 * delayed[i-2]) / a0
        phaser_out = delayed
    phaser_out = signal * (1 - phaser_mix) + phaser_out * phaser_mix
    
    # Flanger: Delay with LFO-modulated time
    flanger_lfo = np.sin(2 * np.pi * flanger_rate * t)
    delay_buffer = np.zeros(total_samples + int(flanger_delay_base * sample_rate + flanger_delay_depth * sample_rate + 1))
    delay_buffer[:total_samples] = signal
    flanger_out = signal.copy()
    for i in range(total_samples):
        delay_time = flanger_delay_base + flanger_delay_depth * flanger_lfo[i]
        delay_pos = i - int(delay_time * sample_rate)
        delayed = delay_buffer[delay_pos] if delay_pos >= 0 and delay_pos < len(delay_buffer) else 0
        flanger_out[i] = signal[i] + delayed * 1.5  # Boosted feedback for stronger comb
    flanger_out = signal * (1 - flanger_mix) + flanger_out * flanger_mix
    
    # Combine phaser and flanger
    wet_signal = (phaser_out + flanger_out) / 2.0  # Average for balance
    output = signal * (1 - wet_mix) + wet_signal * wet_mix * wet_gain  # Boost wet signal
    
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
    
    # Apply phaser/flanger effect
    effected = phaser_flanger_effect(data, sample_rate=44100, phaser_rate=0.5, phaser_depth=0.8, 
                                     phaser_stages=4, flanger_delay_base=0.005, flanger_delay_depth=0.005, 
                                     flanger_rate=0.25, phaser_mix=0.7, flanger_mix=0.7, wet_mix=0.7)
    
    # Save output
    wavfile.write("phaser_flanger_effect.wav", 44100, effected.astype(np.float32))
    print(f"Phaser/flanger effect audio saved to phaser_flanger_effect.wav")