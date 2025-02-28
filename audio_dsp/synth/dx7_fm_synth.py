import numpy as np
import soundfile as sf

class DX7FMSynth:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.algorithm = 1  # Default: linear stack (1–5)
        
        # Operator frequencies ratios (relative to base freq)
        self.freq_ratios = [1.0, 2.0, 1.0, 0.5]  # Op1–Op4
        
        # Modulation indices (strength of modulation)
        self.mod_indices = [1.0, 1.0, 1.0, 1.0]  # Op1–Op4
        
        # Feedback for Op4 (0–5 range, like DX7)
        self.feedback = 0.0
        
        # ADSR for each operator (default values)
        self.adsr = [
            {'attack': 0.01, 'decay': 0.1, 'sustain': 0.7, 'release': 0.2},
            {'attack': 0.01, 'decay': 0.1, 'sustain': 0.7, 'release': 0.2},
            {'attack': 0.01, 'decay': 0.1, 'sustain': 0.7, 'release': 0.2},
            {'attack': 0.01, 'decay': 0.1, 'sustain': 0.7, 'release': 0.2}
        ]

    def apply_adsr(self, duration, attack, decay, sustain, release):
        total_samples = int(self.sample_rate * duration)
        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        sustain_level = sustain

        envelope = np.ones(total_samples)
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, sustain_level, decay_samples)
        sustain_end = total_samples - release_samples
        envelope[attack_samples + decay_samples:sustain_end] = sustain_level
        envelope[sustain_end:] = np.linspace(sustain_level, 0, release_samples)
        return envelope

    def synthesize(self, freq, duration, algorithm=None):
        if algorithm is not None:
            self.algorithm = algorithm
        if not 1 <= self.algorithm <= 5:
            raise ValueError("Algorithm must be 1–5")
        
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        output = np.zeros_like(t)
        
        # Generate envelopes for all operators
        envs = [self.apply_adsr(duration, op['attack'], op['decay'], op['sustain'], op['release'])
                for op in self.adsr]
        
        # Operator signals (sine waves)
        ops = [np.sin(2 * np.pi * freq * ratio * t) for ratio in self.freq_ratios]
        
        # Apply modulation based on algorithm
        if self.algorithm == 1:  # 4 → 3 → 2 → 1
            op4 = ops[3] + self.feedback * ops[3]  # Feedback on Op4
            ops[3] = np.sin(2 * np.pi * freq * self.freq_ratios[3] * t + self.mod_indices[3] * op4)
            ops[2] = np.sin(2 * np.pi * freq * self.freq_ratios[2] * t + self.mod_indices[2] * ops[3])
            ops[1] = np.sin(2 * np.pi * freq * self.freq_ratios[1] * t + self.mod_indices[1] * ops[2])
            output = ops[0] * envs[0]  # Op1 is the carrier
        
        elif self.algorithm == 2:  # (4 → 3) + (2 → 1) → Mix
            op4 = ops[3] + self.feedback * ops[3]
            ops[3] = np.sin(2 * np.pi * freq * self.freq_ratios[3] * t + self.mod_indices[3] * op4)
            ops[2] = np.sin(2 * np.pi * freq * self.freq_ratios[2] * t + self.mod_indices[2] * ops[3])
            ops[1] = np.sin(2 * np.pi * freq * self.freq_ratios[1] * t + self.mod_indices[1] * ops[2])
            output = (ops[2] * envs[2] + ops[0] * envs[0]) * 0.5
        
        elif self.algorithm == 3:  # (4 + 3) → 2 → 1
            op4 = ops[3] + self.feedback * ops[3]
            ops[3] = np.sin(2 * np.pi * freq * self.freq_ratios[3] * t + self.mod_indices[3] * op4)
            ops[2] = np.sin(2 * np.pi * freq * self.freq_ratios[2] * t + self.mod_indices[2] * (ops[3] + ops[2]))
            ops[1] = np.sin(2 * np.pi * freq * self.freq_ratios[1] * t + self.mod_indices[1] * ops[2])
            output = ops[0] * envs[0]
        
        elif self.algorithm == 4:  # 4 → (3 + 2) → 1
            op4 = ops[3] + self.feedback * ops[3]
            ops[3] = np.sin(2 * np.pi * freq * self.freq_ratios[3] * t + self.mod_indices[3] * op4)
            ops[2] = np.sin(2 * np.pi * freq * self.freq_ratios[2] * t + self.mod_indices[2] * ops[3])
            ops[1] = np.sin(2 * np.pi * freq * self.freq_ratios[1] * t + self.mod_indices[1] * ops[3])
            output = ops[0] * envs[0] + ops[1] * envs[1] * 0.5
        
        elif self.algorithm == 5:  # All independent
            op4 = ops[3] + self.feedback * ops[3]
            ops[3] = np.sin(2 * np.pi * freq * self.freq_ratios[3] * t + self.mod_indices[3] * op4)
            output = sum(op * env for op, env in zip(ops, envs)) / 4.0
        
        # Normalize to avoid clipping
        output = output / np.max(np.abs(output))
        return output

def save_dx7_sound(filename, freq=110, duration=1.0, algorithm=1, sample_rate=44100):
    synth = DX7FMSynth(sample_rate=sample_rate)
    synth.algorithm = algorithm
    audio = synth.synthesize(freq, duration)
    sf.write(filename, audio, sample_rate, subtype='PCM_16')
    print(f"Sound saved to {filename}")

# Test it
if __name__ == "__main__":
    # Test all algorithms
    for algo in range(1, 6):
        save_dx7_sound(f"dx7_algo{algo}.wav", freq=110, duration=2.0, algorithm=algo)