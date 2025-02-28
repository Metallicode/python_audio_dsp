import numpy as np
import soundfile as sf

class DrumSynth:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def _timevector(self, length):
        return np.linspace(0, length, int(self.sample_rate * length), endpoint=False)

    def _norm(self, signal):
        max_amp = np.max(np.abs(signal))
        return signal / max_amp if max_amp > 0 else signal

    def _make_signal(self, shape, frequency, length, fm_depth=0.0):
        """Generate signal with shape: sine, square, fm, or noise types."""
        t = self._timevector(length)
        if isinstance(frequency, (int, float)):
            frequency = np.full_like(t, frequency)
        
        if shape == "sine":
            return np.sin(2 * np.pi * np.cumsum(frequency) / self.sample_rate)
        elif shape == "square":
            return np.sign(np.sin(2 * np.pi * np.cumsum(frequency) / self.sample_rate))
        elif shape == "fm":
            modulator = np.sin(2 * np.pi * np.cumsum(frequency * 2) / self.sample_rate)
            return np.sin(2 * np.pi * np.cumsum(frequency) / self.sample_rate + fm_depth * modulator)
        elif shape in ["normal_noise", "random_noise", "pink_noise"]:
            samples = int(self.sample_rate * length)
            if shape == "normal_noise":
                return np.random.normal(0, 1, samples)
            elif shape == "random_noise":
                return np.random.uniform(-1, 1, samples)
            elif shape == "pink_noise":
                white = np.random.normal(0, 1, samples)
                return np.convolve(white, np.ones(10)/10, mode='same')
        else:
            print(f"Warning: Invalid shape '{shape}'—defaulting to sine")
            return np.sin(2 * np.pi * np.cumsum(frequency) / self.sample_rate)

    def _log_envelope(self, t, decay_factor):
        return np.exp(-decay_factor * t / t[-1]) if len(t) > 1 else np.ones_like(t)

    def _filter(self, signal, filter_type="high", cutoff=1000):
        freqs = np.fft.fftfreq(len(signal), 1/self.sample_rate)
        fft = np.fft.fft(signal)
        if filter_type == "high":
            fft[np.abs(freqs) < cutoff] = 0
        elif filter_type == "low":
            fft[np.abs(freqs) > cutoff] = 0
        elif filter_type == "band":
            fft[(np.abs(freqs) < cutoff[0]) | (np.abs(freqs) > cutoff[1])] = 0
        return np.real(np.fft.ifft(fft))

    def kick(self, length=1.0, max_pitch=1000, min_pitch=50, decay_factor=50, signal_mix="sine:1.0"):
        """Kick drum with pitch sweep and signal mix."""
        t = self._timevector(length)
        t = t + (1.0 - max(t))
        freq_sweep = (max_pitch - min_pitch) * (t ** decay_factor) + min_pitch
        
        mix_dict = {k.split(':')[0].strip(): float(k.split(':')[1]) for k in signal_mix.split(',')}
        total_weight = sum(mix_dict.values())
        mix_dict = {k: v / total_weight for k, v in mix_dict.items()}
        print(f"Kick signal mix: {mix_dict}")
        
        kick = np.zeros_like(t)
        for shape, weight in mix_dict.items():
            kick += weight * self._make_signal(shape, freq_sweep, length, fm_depth=0.5 if shape == "fm" else 0.0)
        kick = kick[::-1]
        return self._norm(kick)

    def snare(self, length=1.0, high_pitch=800, low_pitch=250, decay_factor=50, mix=0.5, signal_mix="sine:1.0"):
        """Snare with noise and tonal layers."""
        noise = self._make_signal("normal_noise", 0, length)  # Freq=0 for noise
        t = self._timevector(length)
        
        mix_dict = {k.split(':')[0].strip(): float(k.split(':')[1]) for k in signal_mix.split(',')}
        total_weight = sum(mix_dict.values())
        mix_dict = {k: v / total_weight for k, v in mix_dict.items()}
        print(f"Snare signal mix: {mix_dict}")
        
        tone = np.zeros_like(t)
        for shape, weight in mix_dict.items():
            if shape == "fm":
                tone += weight * self._make_signal("fm", high_pitch, length, fm_depth=1.0)
            else:
                tone += weight * (self._make_signal(shape, high_pitch, length) + 
                                 self._make_signal(shape, low_pitch, length))
        sner_tone = (noise * (1.0 - mix)) + (tone * mix)
        env = self._log_envelope(t, decay_factor)
        return self._norm(sner_tone * env)

    def cymbal(self, length=1.0, op_a_freq=4000, op_b_freq=400, noise_env=40, tone_env=10, 
               cutoff=3000, mix=0.5, signal_mix="fm:1.0"):
        """Cymbal with FM tone and noise."""
        t = self._timevector(length)
        n_env = self._log_envelope(t, noise_env)
        t_env = self._log_envelope(t, tone_env)
        
        noise = self._make_signal("random_noise", 0, length) * n_env
        
        mix_dict = {k.split(':')[0].strip(): float(k.split(':')[1]) for k in signal_mix.split(',')}
        total_weight = sum(mix_dict.values())
        mix_dict = {k: v / total_weight for k, v in mix_dict.items()}
        print(f"Cymbal signal mix: {mix_dict}")
        
        tone = np.zeros_like(t)
        for shape, weight in mix_dict.items():
            if shape == "fm":
                tone += weight * self._make_signal("fm", op_b_freq, length, fm_depth=op_a_freq/op_b_freq)
            else:
                tone += weight * self._make_signal(shape, op_b_freq, length)
        tone = tone[::-1] * t_env
        
        cymbal = (noise * (1.0 - mix)) + (tone * mix)
        cymbal = self._filter(cymbal, "high", cutoff)
        return self._norm(cymbal)

    def clap(self, length=0.2, pitch=1500, burst_count=3, decay_factor=20, noise_mix=0.7, signal_mix="square:1.0"):
        """Clap with noise bursts and tonal slap."""
        t = self._timevector(length)
        env = self._log_envelope(t, decay_factor)
        
        noise = self._make_signal("random_noise", 0, length)
        burst_env = np.zeros_like(t)
        burst_len = int(length / burst_count * self.sample_rate)
        for i in range(burst_count):
            start = i * burst_len // 2
            end = min(start + burst_len, len(t))
            burst_env[start:end] = np.linspace(1, 0, end - start)
        noise = noise * burst_env * noise_mix
        
        mix_dict = {k.split(':')[0].strip(): float(k.split(':')[1]) for k in signal_mix.split(',')}
        total_weight = sum(mix_dict.values())
        mix_dict = {k: v / total_weight for k, v in mix_dict.items()}
        print(f"Clap signal mix: {mix_dict}")
        
        tone = np.zeros_like(t)
        for shape, weight in mix_dict.items():
            tone += weight * self._make_signal(shape, pitch, length)
        clap = noise + (tone * (1 - noise_mix))
        clap *= env
        return self._norm(clap)

    def rim(self, length=0.1, pitch=2000, decay_factor=30, noise_mix=0.3, signal_mix="fm:1.0"):
        """Rim shot with click and short tone."""
        t = self._timevector(length)
        env = self._log_envelope(t, decay_factor)
        
        click = self._make_signal("pink_noise", 0, length) * noise_mix
        
        mix_dict = {k.split(':')[0].strip(): float(k.split(':')[1]) for k in signal_mix.split(',')}
        total_weight = sum(mix_dict.values())
        mix_dict = {k: v / total_weight for k, v in mix_dict.items()}
        print(f"Rim signal mix: {mix_dict}")
        
        tone = np.zeros_like(t)
        for shape, weight in mix_dict.items():
            tone += weight * self._make_signal(shape, pitch, length, fm_depth=0.5)
        rim = (click + tone * (1 - noise_mix)) * env
        rim = self._filter(rim, "high", 1000)
        return self._norm(rim)

    def tom(self, length=0.5, max_pitch=300, min_pitch=80, decay_factor=20, noise_mix=0.1, signal_mix="sine:1.0"):
        """Tom with pitch sweep and subtle noise."""
        t = self._timevector(length)
        env = self._log_envelope(t, decay_factor)
        
        freq_sweep = (max_pitch - min_pitch) * (t ** decay_factor) + min_pitch
        
        mix_dict = {k.split(':')[0].strip(): float(k.split(':')[1]) for k in signal_mix.split(',')}
        total_weight = sum(mix_dict.values())
        mix_dict = {k: v / total_weight for k, v in mix_dict.items()}
        print(f"Tom signal mix: {mix_dict}")
        
        tone = np.zeros_like(t)
        for shape, weight in mix_dict.items():
            if shape == "fm":
                tone += weight * self._make_signal("fm", freq_sweep[-1], length, fm_depth=0.2)
            else:
                tone += weight * self._make_signal(shape, freq_sweep, length)
        tone = tone * (1 - noise_mix)
        
        noise = self._make_signal("pink_noise", 0, length) * noise_mix
        tom = (tone + noise) * env
        tom = self._filter(tom, "low", 500)
        return self._norm(tom)

# Test it
if __name__ == "__main__":
    synth = DrumSynth()
    
    drums = {
        "kick": synth.kick(length=0.5, max_pitch=1000, min_pitch=50,decay_factor=80, signal_mix="fm:0.7, sine:0.3"),
        "snare": synth.snare(length=0.3, high_pitch=200, low_pitch=70, decay_factor=7, signal_mix="fm:1.0"),
        "cymbal": synth.cymbal(length=1.0, op_a_freq=4000, op_b_freq=400, signal_mix="fm:0.8, square:0.2"),
        "clap": synth.clap(length=0.7, pitch=600, signal_mix="square:1.0"),
        "rim": synth.rim(length=0.3, pitch=2000, signal_mix="fm:0.6, square:0.4"),
        "tom": synth.tom(length=0.5, max_pitch=300, min_pitch=80, signal_mix="sine:0.5, square:0.5")
    }
    
    for name, signal in drums.items():
        sf.write(f"{name}.wav", signal, synth.sample_rate, subtype='PCM_16')
        print(f"Saved {name}.wav")