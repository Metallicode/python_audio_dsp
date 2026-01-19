import numpy as np
from scipy import interpolate


def _pitch_drift(input_signal, sample_rate, drift_depth=0.1, drift_rate=0.05):
    """Apply a slow, random pitch drift to an input signal efficiently."""
    signal = np.array(input_signal, dtype=np.float64)
    signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else signal
    total_samples = len(signal)
    t = np.linspace(0, total_samples / sample_rate, total_samples, endpoint=False)
    lfo = np.random.normal(0, 1, total_samples)
    lfo = np.convolve(lfo, np.ones(500)/500, mode='same')
    pitch_shift = drift_depth * np.sin(2 * np.pi * drift_rate * t + lfo)
    speed = 2 ** (pitch_shift / 12.0)
    warped_time = np.cumsum(1.0 / speed)
    warped_time = warped_time * (total_samples / warped_time[-1])
    interp_func = interpolate.interp1d(np.arange(total_samples), signal, kind='linear',
                                       bounds_error=False, fill_value=0)
    output = interp_func(warped_time)
    return output / np.max(np.abs(output)) if np.max(np.abs(output)) > 0 else output


def _flutter(input_signal, sample_rate, base_rate=5.0, rate_diff=1.0, depth=0.05):
    """Apply a flutter effect with dual-sine LFO modulation."""
    signal = np.array(input_signal, dtype=np.float64)
    signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else signal
    total_samples = len(signal)
    t = np.linspace(0, total_samples / sample_rate, total_samples, endpoint=False)
    lfo1 = np.sin(2 * np.pi * base_rate * t)
    lfo2 = np.sin(2 * np.pi * (base_rate + rate_diff) * t)
    lfo = (lfo1 + lfo2) / 2.0
    speed = 1.0 + (depth * lfo)
    warped_time = np.cumsum(1.0 / speed)
    warped_time = warped_time * (total_samples / warped_time[-1])
    interp_func = interpolate.interp1d(np.arange(total_samples), signal, kind='linear',
                                       bounds_error=False, fill_value=0)
    output = interp_func(warped_time)
    return output / np.max(np.abs(output)) if np.max(np.abs(output)) > 0 else output


def _lowpass_filter(signal, sample_rate, cutoff=3000):
    """Apply FFT-based low-pass filter."""
    freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
    fft = np.fft.fft(signal)
    taper = np.exp(-((np.abs(freqs) - cutoff) / (cutoff * 0.5))**2)
    fft *= taper
    fft[np.abs(freqs) > cutoff * 1.5] = 0
    return np.real(np.fft.ifft(fft))


def _tape_saturation(signal, drive=2.0, warmth=0.1):
    """Apply tape saturation effect."""
    driven_signal = signal * drive
    saturated = np.tanh(driven_signal) + warmth * (driven_signal ** 3)
    return 0.5 * signal + 0.5 * saturated


def _simple_compressor(signal, threshold=0.7, ratio=4.0):
    """Apply simple compression."""
    output = signal.copy()
    for i in range(len(signal)):
        if abs(signal[i]) > threshold:
            gain = threshold + (abs(signal[i]) - threshold) / ratio
            output[i] = np.sign(signal[i]) * gain if signal[i] != 0 else 0
    max_amp = np.max(np.abs(output))
    return output / max_amp if max_amp > 0 else output


def delay(input_signal, sample_rate=44100, delay_time=0.25, feedback=0.7, mix=0.5,
          mode="digital", lp_cutoff=3000, flutter_base_rate=5.0, flutter_depth=0.005,
          pitch_drift_depth=0.1, drive=2.0, threshold=0.7, ratio=4.0):
    """
    Apply a delay effect to an input signal.

    Args:
        input_signal: Input audio array (mono, normalized to ±1)
        sample_rate: Sample rate in Hz (default 44100)
        delay_time: Delay time in seconds (default 0.25)
        feedback: Feedback amount 0–1 (default 0.7)
        mix: Wet/dry mix 0–1 (default 0.5)
        mode: "digital" for clean delay, "analog" for tape-style delay
        lp_cutoff: Low-pass filter cutoff for analog mode (default 3000 Hz)
        flutter_base_rate: Flutter LFO rate for analog mode (default 5.0 Hz)
        flutter_depth: Flutter depth for analog mode (default 0.005)
        pitch_drift_depth: Pitch drift amount for analog mode (default 0.1)
        drive: Saturation drive for analog mode (default 2.0)
        threshold: Compressor threshold for analog mode (default 0.7)
        ratio: Compressor ratio for analog mode (default 4.0)

    Returns:
        Output audio array with delay effect applied
    """
    # Ensure input is float64 and normalized
    audio = np.array(input_signal, dtype=np.float64)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp

    total_samples = len(audio)
    delay_samples = int(delay_time * sample_rate)
    output = np.zeros(total_samples)
    delayed_buffer = np.zeros(total_samples + delay_samples)

    # Process raw delay into delayed_buffer
    for i in range(total_samples):
        delay_pos = i - delay_samples
        delayed = delayed_buffer[delay_pos] if delay_pos >= 0 and delay_pos < len(delayed_buffer) else 0
        delayed_buffer[i] = audio[i] + feedback * delayed

    # Apply analog effects to delayed signal
    wet_signal = delayed_buffer[:total_samples]
    if mode == "analog":
        wet_signal = _pitch_drift(wet_signal, sample_rate, drift_depth=pitch_drift_depth, drift_rate=0.05)
        wet_signal = _flutter(wet_signal, sample_rate, base_rate=flutter_base_rate, rate_diff=1.0,
                              depth=flutter_depth)
        wet_signal = _lowpass_filter(wet_signal, sample_rate, cutoff=lp_cutoff)
        wet_signal = _tape_saturation(wet_signal, drive=drive, warmth=0.1)
        wet_signal = _simple_compressor(wet_signal, threshold=threshold, ratio=ratio)

    # Mix dry and wet
    for i in range(total_samples):
        output[i] = audio[i] * (1 - mix) + wet_signal[i] * mix

    # Normalize
    max_amp = np.max(np.abs(output))
    if max_amp > 0:
        output = output / max_amp

    return output


# Backward compatibility: keep SuperDelay class
class SuperDelay:
    """Legacy class for backward compatibility. Use delay() function instead."""

    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def pitch_drift(self, input_signal, drift_depth=0.1, drift_rate=0.05):
        return _pitch_drift(input_signal, self.sample_rate, drift_depth, drift_rate)

    def flutter_effect(self, input_signal, base_rate=5.0, rate_diff=1.0, depth=0.05):
        return _flutter(input_signal, self.sample_rate, base_rate, rate_diff, depth)

    def _lowpass_filter(self, signal, cutoff=3000):
        return _lowpass_filter(signal, self.sample_rate, cutoff)

    def _tape_saturation(self, signal, drive=2.0, warmth=0.1):
        return _tape_saturation(signal, drive, warmth)

    def _simple_compressor(self, signal, threshold=0.7, ratio=4.0):
        return _simple_compressor(signal, threshold, ratio)

    def delay(self, input_file, output_file, delay_time=0.25, feedback=0.7, mix=0.5,
              mode="digital", lp_cutoff=3000, flutter_base_rate=5.0, flutter_depth=0.005,
              pitch_drift_depth=0.1, drive=2.0, resonance=0.7, q=1.0, threshold=0.7, ratio=4.0):
        """Legacy file-based delay. Use delay() function for array-based processing."""
        from audio_dsp.utils import wav_io as wavfile
        import librosa

        # Load audio
        samplerate, audio = wavfile.read(input_file)
        if samplerate != self.sample_rate:
            audio = librosa.resample(audio.astype(np.float64), orig_sr=samplerate, target_sr=self.sample_rate)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio / np.max(np.abs(audio))

        # Process using the new array-based function
        output = delay(audio, self.sample_rate, delay_time, feedback, mix, mode, lp_cutoff,
                       flutter_base_rate, flutter_depth, pitch_drift_depth, drive, threshold, ratio)

        # Save
        wavfile.write(output_file, self.sample_rate, output.astype(np.float32))
        print(f"Delayed audio saved to {output_file}")


# Test it
if __name__ == "__main__":
    from audio_dsp.utils import wav_io as wavfile
    import librosa

    # Load sample data
    samplerate, data = wavfile.read("input.wav")
    if samplerate != 44100:
        data = librosa.resample(data.astype(np.float64), orig_sr=samplerate, target_sr=44100)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data / np.max(np.abs(data))

    # Digital mode: Clean delay
    digital_delay = delay(data, sample_rate=44100, delay_time=0.25, feedback=0.5, mix=0.5,
                          mode="digital")
    wavfile.write("delay_digital.wav", 44100, digital_delay.astype(np.float32))
    print("Digital delay saved to delay_digital.wav")

    # Analog mode: Tape echo
    analog_delay = delay(data, sample_rate=44100, delay_time=0.05, feedback=0.8, mix=0.7,
                         mode="analog", lp_cutoff=6000, flutter_base_rate=5.0, flutter_depth=0.01,
                         pitch_drift_depth=0.1, drive=6.0, threshold=0.9, ratio=4.0)
    wavfile.write("delay_analog.wav", 44100, analog_delay.astype(np.float32))
    print("Analog delay saved to delay_analog.wav")
