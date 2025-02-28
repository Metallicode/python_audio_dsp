import numpy as np
import soundfile as sf

class SuperStackedSynth:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.wave_shapes = ["sine", "triangle", "saw", "square"]

    def generate_oscillator(self, wave_type, freq, t):
        """Generate a single oscillator waveform."""
        if wave_type == "sine":
            return np.sin(2 * np.pi * freq * t)
        elif wave_type == "triangle":
            return 2 * np.abs(2 * (t * freq % 1) - 1) - 1
        elif wave_type == "saw":
            return 2 * (t * freq % 1) - 1
        elif wave_type == "square":
            return np.sign(np.sin(2 * np.pi * freq * t))
        else:
            raise ValueError("Invalid wave_type. Use: sine, triangle, saw, square")

    def synthesize(self, base_freq, duration, num_oscillators=100, detune_spread=0.02, 
                  wave_mix=None, output_file="super_synth.wav", fade_in_time=0.1):
        """
        Synthesize a stacked super-synth sound with balanced fade-in.
        - base_freq: Central frequency in Hz
        - duration: Length in seconds
        - num_oscillators: Number of stacked oscillators (1–10000)
        - detune_spread: Max detuning factor (e.g., 0.02 = ±2%)
        - wave_mix: Dict of wave shape weights (default equal mix)
        - output_file: Output WAV file path
        - fade_in_time: Fade-in duration per oscillator (seconds, default 0.1)
        """
        if not 1 <= num_oscillators <= 10000:
            raise ValueError("num_oscillators must be 1–10000")
        
        # Time array
        total_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, total_samples, endpoint=False)
        print(f"Generating {total_samples} samples ({duration}s)")
        
        # Wave mix setup
        if wave_mix is None:
            wave_mix = {shape: 1.0 / len(self.wave_shapes) for shape in self.wave_shapes}
        total_weight = sum(wave_mix.values())
        wave_mix = {k: v / total_weight for k, v in wave_mix.items()}
        
        # Detuning array
        detune_factors = 1 + np.random.uniform(-detune_spread, detune_spread, num_oscillators)
        freqs = base_freq * detune_factors
        print(f"Detuning range: {np.min(freqs):.2f} Hz to {np.max(freqs):.2f} Hz")
        
        # Fade-in envelope per oscillator (staggered start)
        fade_samples = int(fade_in_time * self.sample_rate)
        fade_in = np.ones(total_samples)
        if fade_samples < total_samples:
            fade_in[:fade_samples] = np.linspace(0, 1, fade_samples)
        
        # Generate stacked oscillators
        output = np.zeros(total_samples)
        for i in range(num_oscillators):
            osc = np.zeros(total_samples)
            for wave_type, weight in wave_mix.items():
                osc += weight * self.generate_oscillator(wave_type, freqs[i], t)
            # Apply fade-in scaled by steady-state level
            osc *= fade_in * (1.0 / np.sqrt(num_oscillators))  # RMS-like scaling
            output += osc
        
        # Normalize
        output = output / np.max(np.abs(output))
        print(f"Output max amp: {np.max(np.abs(output)):.5f}")
        
        # Debug RMS over time
        rms = np.sqrt(np.mean(output[:int(self.sample_rate/10)]**2))  # First 0.1s
        rms_mid = np.sqrt(np.mean(output[int(self.sample_rate):int(self.sample_rate*2)]**2))  # Middle 1s
        print(f"RMS: Start {rms:.5f}, Mid {rms_mid:.5f}")
        
        # Save
        sf.write(output_file, output, self.sample_rate, subtype='PCM_16')
        print(f"Sound saved to {output_file}")

# Test it
if __name__ == "__main__":
    synth = SuperStackedSynth()
    
    # Basic stack: 100 oscillators
    synth.synthesize(base_freq=110, duration=4.0, num_oscillators=100, detune_spread=0.02, 
                     output_file="super_synth_100.wav", fade_in_time=0.1)
    
    # Ridiculous stack: 5000 oscillators
    synth.synthesize(base_freq=220, duration=4.0, num_oscillators=5000, detune_spread=0.05, 
                     wave_mix={"sine": 0.1, "triangle": 0.1, "saw": 0.7, "square": 0.1}, 
                     output_file="super_synth_5000.wav", fade_in_time=0.2)
    
    # Max stack: 10000 oscillators
    synth.synthesize(base_freq=440, duration=4.0, num_oscillators=10000, detune_spread=0.03, 
                     wave_mix={"square": 1.0}, output_file="super_synth_10000.wav", fade_in_time=0.3)