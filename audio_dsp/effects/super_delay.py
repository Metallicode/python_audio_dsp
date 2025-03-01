import numpy as np
import soundfile as sf
import librosa
from scipy.io import wavfile

class SuperDelay:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def _lowpass_filter(self, signal, cutoff):
        """Applies a smooth low-pass filter using FFT."""
        freqs = np.fft.fftfreq(len(signal), 1/self.sample_rate)
        fft = np.fft.fft(signal)
        taper = np.exp(-((np.abs(freqs) - cutoff) / (cutoff * 0.5))**2)
        fft *= taper
        fft[np.abs(freqs) > cutoff * 1.5] = 0
        return np.real(np.fft.ifft(fft))

    def _soft_clip(self, signal, gain=1.0):
        """Applies soft-clipping for analog saturation effect."""
        return np.tanh(signal * gain + 0.05 * signal**3) / gain

    def _smooth_lfo(self, length, rate, depth):
        """Generate a smooth stochastic LFO."""
        t = np.linspace(0, length / self.sample_rate, length, endpoint=False)
        noise = np.random.normal(0, 1, length)
        noise = np.convolve(noise, np.ones(200)/200, mode='same')  # Smooth over 200 samples
        return depth * noise * np.sin(2 * np.pi * rate * t)  # Slow modulation

    def delay(self, input_file, output_file, delay_time=0.25, feedback=0.5, mix=0.5, 
              mode="digital", lp_cutoff=1500, flutter_rate=0.5, flutter_depth=0.005, 
              pitch_drift=0.1, timing_jitter=0.05, hiss_level=0.0001):
        """Applies digital or analog delay to an audio file."""
        
        # Load audio
        audio, sr = sf.read(input_file)
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio / np.max(np.abs(audio))  # Normalize input

        total_samples = len(audio)
        delay_samples = int(delay_time * self.sample_rate)
        output = np.zeros(total_samples)
        buffer = np.zeros(total_samples + delay_samples)
        
        # Hiss for analog mode
        hiss = np.zeros(total_samples)
        if mode == "analog" and hiss_level > 0:
            hiss = np.random.normal(0, 1, total_samples)
            hiss = np.convolve(hiss, np.ones(50)/50, mode='same')
            hiss = self._lowpass_filter(hiss, 500) * hiss_level  # Darker hiss
        
        # Slow stochastic LFOs for flutter and pitch drift
        if mode == "analog":
            flutter_lfo = self._smooth_lfo(total_samples, flutter_rate, flutter_depth)
            pitch_lfo = self._smooth_lfo(total_samples, 0.05, pitch_drift)  # Slower 0.05 Hz drift

        # Process signal
        for i in range(total_samples):
            # Compute delay position
            delay_pos = i - delay_samples
            if mode == "analog":
                flutter = flutter_lfo[i]
                jitter = np.random.uniform(-timing_jitter, timing_jitter)
                drift = pitch_lfo[i]
                # delay_pos = i - int((delay_time + flutter + jitter + drift) * self.sample_rate)
                delay_pos = i - int((delay_time  + jitter + drift) * self.sample_rate)

                delay_pos = max(0, min(delay_pos, len(buffer) - 1))  # Clamp
            
            # Fetch delayed signal
            delayed = buffer[delay_pos] if delay_pos >= 0 else 0.0
            if mode == "analog":
                delayed_chunk = np.array([delayed] * int(0.01 * self.sample_rate))
                delayed = self._lowpass_filter(delayed_chunk, lp_cutoff)[0]
                delayed = self._soft_clip(delayed, gain=1.0)
            
            # Mix output with boosted wet signal
            output[i] = audio[i] * (1 - mix) + (delayed * 2.0) * mix + hiss[i] * (1 - mix)
            
            # Update buffer with feedback
            buffer[i] = audio[i] + feedback * delayed
            
            # Log every 0.25s
            if i % delay_samples < 10:
                print(f"Sample {i}: Input {audio[i]:.5f}, DelayPos {delay_pos}, Delayed {delayed:.5f}, Buffer[{i}] {buffer[i]:.5f}, Output {output[i]:.5f}")

        # Normalize
        max_amp = np.max(np.abs(output))
        if max_amp > 0:
            output = output / max_amp
        print(f"Output max amp: {max_amp:.5f} (post-norm: {np.max(np.abs(output)):.5f})")

        # Save
        sf.write(output_file, output, self.sample_rate, subtype='PCM_16')
        print(f"Delayed audio saved to {output_file}")

# Test it
if __name__ == "__main__":
    delay = SuperDelay()
    
    # Digital mode: Clean delay
    delay.delay("input.wav", "delay_digital.wav", delay_time=0.25, feedback=0.5, mix=0.5, 
                mode="digital", lp_cutoff=5000)
    
    # Analog mode: Tape echo
    delay.delay("input.wav", "delay_analog.wav", delay_time=0.25, feedback=0.7, mix=0.9, 
                mode="analog", lp_cutoff=1000, flutter_rate=0.5, flutter_depth=0.005, 
                pitch_drift=0.1, timing_jitter=0.05, hiss_level=0.0001)