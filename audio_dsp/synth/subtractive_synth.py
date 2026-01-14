import numpy as np
import soundfile as sf

class SubtractiveSynth:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        
        # Default ADSR values
        self.attack = 0.01
        self.decay = 0.1
        self.sustain = 0.7
        self.release = 0.2
        
        # Default Filter Parameters
        self.filter_type = "lowpass"  # 'lowpass', 'highpass', 'bandpass'
        self.filter_cutoff = 1000  # Hz
        self.filter_resonance = 0.7  # Resonance amplitude (0–5)
        self.filter_q = 1.0  # Q factor (0.1–10)

        # Default LFOs
        self.lfo_freq = 5
        self.lfo_depth = 0.2
        self.lfo_target = "filter"
        
        # Default waveform
        self.osc_wave = "saw"
        self.pwm_depth = 0.5
        self.wavetable = None

    def generate_waveform(self, wave_type, freq, duration):
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        if wave_type == "sine":
            return np.sin(2 * np.pi * freq * t)
        elif wave_type == "square":
            return np.sign(np.sin(2 * np.pi * freq * t))
        elif wave_type == "saw":
            return 2 * (t * freq % 1) - 1
        elif wave_type == "triangle":
            return 2 * np.abs(2 * (t * freq % 1) - 1) - 1
        elif wave_type == "pwm":
            duty_cycle = 0.5 + self.pwm_depth * np.sin(2 * np.pi * freq * t)
            return np.where((t * freq) % 1 < duty_cycle, 1, -1)
        elif wave_type == "wavetable" and self.wavetable is not None:
            wavetable_size = len(self.wavetable)
            indices = (t * freq * wavetable_size) % wavetable_size
            return np.interp(indices, np.arange(wavetable_size), self.wavetable)
        else:
            raise ValueError("Invalid waveform type.")

    def generate_lfo(self, duration):
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        return np.sin(2 * np.pi * self.lfo_freq * t) * self.lfo_depth

    def apply_adsr(self, signal, attack, decay, sustain, release):
        total_samples = len(signal)
        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        sustain_level = sustain

        # Scale ADSR stages if they exceed total duration
        total_adsr = attack_samples + decay_samples + release_samples
        if total_adsr > total_samples:
            scale = total_samples / total_adsr
            attack_samples = int(attack_samples * scale)
            decay_samples = int(decay_samples * scale)
            release_samples = int(release_samples * scale)

        envelope = np.ones(total_samples)

        # Attack phase
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

        # Decay phase
        decay_end = min(attack_samples + decay_samples, total_samples)
        actual_decay = decay_end - attack_samples
        if actual_decay > 0:
            envelope[attack_samples:decay_end] = np.linspace(1, sustain_level, actual_decay)

        # Sustain phase
        sustain_end = max(0, total_samples - release_samples)
        if sustain_end > decay_end:
            envelope[decay_end:sustain_end] = sustain_level

        # Release phase
        if release_samples > 0 and sustain_end < total_samples:
            actual_release = total_samples - sustain_end
            envelope[sustain_end:] = np.linspace(sustain_level, 0, actual_release)

        return signal * envelope

    def apply_filter(self, signal, cutoff, resonance, q, filter_type="lowpass"):
        """
        Enhanced 4-pole resonant filter with adjustable Q and type.
        - signal: Input audio array
        - cutoff: Array of cutoff frequencies in Hz (50–5000)
        - resonance: Resonance amplitude (0–5)
        - q: Q factor (0.1–10)
        - filter_type: 'lowpass', 'highpass', 'bandpass'
        """
        nyquist = self.sample_rate / 2
        cutoff = np.clip(cutoff, 50, nyquist - 1)
        resonance = np.clip(resonance, 0, 5.0)
        q = np.clip(q, 0.1, 10.0)
        
        res_amplitude = resonance * 0.25 * (1 + np.exp(0.5 * resonance))
        y0, y1, y2, y3 = 0.0, 0.0, 0.0, 0.0
        output = np.zeros_like(signal)
        
        for i in range(len(signal)):
            wc = 2 * np.pi * cutoff[i] / self.sample_rate
            alpha = np.sin(wc) / q
            input_sample = signal[i]
            
            if filter_type == "lowpass":
                feedback = res_amplitude * y3
                y0 = np.tanh(y0 + alpha * (input_sample - feedback - y0))
                y1 = np.tanh(y1 + alpha * (y0 - y1))
                y2 = np.tanh(y2 + alpha * (y1 - y2))
                y3 = np.tanh(y3 + alpha * (y2 - y3))
                y3 *= 0.98
                output[i] = y3
            
            elif filter_type == "highpass":
                # High-pass: Differential ladder
                feedback = res_amplitude * y3
                diff = input_sample - y0  # Differentiate input
                y0 = np.tanh(diff + alpha * (input_sample - y0 - feedback))
                y1 = np.tanh(y1 + alpha * (y0 - y1))
                y2 = np.tanh(y2 + alpha * (y1 - y2))
                y3 = np.tanh(y3 + alpha * (y2 - y3))
                y3 *= 0.98
                output[i] = y0 * 2.0  # Boost highs
            
            elif filter_type == "bandpass":
                feedback = res_amplitude * y3
                y0 = np.tanh(y0 + alpha * (input_sample - feedback - y0))
                y1 = np.tanh(y1 + alpha * (y0 - y1))
                y2 = np.tanh(y2 + alpha * (y1 - y2))
                y3 = np.tanh(y3 + alpha * (y2 - y3))
                y3 *= 0.98
                output[i] = y2 - y3
        
        output = np.tanh(output)  # Final clip
        return output

    def synthesize(self, freq, duration, attack=None, decay=None, sustain=None, release=None):
        attack = attack if attack is not None else self.attack
        decay = decay if decay is not None else self.decay
        sustain = sustain if sustain is not None else self.sustain
        release = release if release is not None else self.release

        signal = self.generate_waveform(self.osc_wave, freq, duration)
        lfo = self.generate_lfo(duration)

        t = np.linspace(0, duration, len(signal))

        if self.lfo_target == "filter":
            cutoff_modulated = self.filter_cutoff * (1 + lfo)
            cutoff_modulated = np.clip(cutoff_modulated, 20, self.sample_rate / 2)
            signal = self.apply_filter(signal, cutoff_modulated, self.filter_resonance, 
                                     self.filter_q, self.filter_type)
        elif self.lfo_target == "pitch":
            freq_modulated = freq * (1 + lfo)
            signal = np.interp(t, t, self.generate_waveform(self.osc_wave, freq_modulated, duration))
        elif self.lfo_target == "amplitude":
            signal *= (1 + lfo)

        signal = self.apply_adsr(signal, attack, decay, sustain, release)
        return signal

# Test it
if __name__ == "__main__":
    s = SubtractiveSynth(sample_rate=44100)
    
    # Low-pass
    wave_lp = s.synthesize(freq=70, duration=4.0, attack=0.02, decay=0.1, sustain=0.6, release=0.3)
    sf.write("tb303_lp.wav", wave_lp, 44100, subtype='PCM_16')
    print("Saved low-pass to tb303_lp.wav")
    
    # High-pass
    s.filter_type = "highpass"
    wave_hp = s.synthesize(freq=70, duration=4.0, attack=0.02, decay=0.1, sustain=0.6, release=0.3)
    sf.write("tb303_hp.wav", wave_hp, 44100, subtype='PCM_16')
    print("Saved high-pass to tb303_hp.wav")
    
    # Bandpass
    s.filter_type = "bandpass"
    wave_bp = s.synthesize(freq=70, duration=4.0, attack=0.02, decay=0.1, sustain=0.6, release=0.3)
    sf.write("tb303_bp.wav", wave_bp, 44100, subtype='PCM_16')
    print("Saved bandpass to tb303_bp.wav")