import numpy as np
import soundfile as sf
import json

class PhysicalModelingSynth:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def synthesize(self, frequency, length, transient_file, spectral_file, output_file, decay_rate=0.5):
        """
        Synthesize sound by exciting spectral body with transient.
        - frequency: Target frequency in Hz
        - length: Desired duration in seconds
        - transient_file: .transient file path
        - spectral_file: .spectral file path
        - output_file: Output WAV file path
        - decay_rate: Body decay speed (seconds, default 0.5)
        """
        # Load transient
        with open(transient_file, 'r') as f:
            transient_data = json.load(f)
        transient = np.array(transient_data["samples"])
        transient_len = len(transient)
        transient = transient / np.max(np.abs(transient))  # Normalize to 1.0
        print(f"Transient loaded: {transient_len} samples ({transient_len/self.sample_rate:.3f}s), max amp: {np.max(np.abs(transient)):.5f}")
        
        # # Save transient for debugging
        # sf.write("transient_debug.wav", transient, self.sample_rate, subtype='PCM_16')
        # print("Saved transient_debug.wav")
        
        # Load spectral data
        with open(spectral_file, 'r') as f:
            spectral_data = json.load(f)
        peaks = spectral_data["peaks"]
        
        # Generate time array
        total_samples = int(length * self.sample_rate)
        t = np.linspace(0, length, total_samples, endpoint=False)
        print(f"Total samples: {total_samples} ({length}s)")
        
        # Resynthesize spectral body
        body = np.zeros(total_samples)
        for peak in peaks:
            freq = peak["frequency"] * (frequency / peaks[0]["frequency"])
            amp = peak["amplitude"]
            phase = peak["phase"]
            body += amp * np.sin(2 * np.pi * freq * t + phase)
        body = body / (len(peaks) * 10)  # Scale down
        print(f"Body max amp pre-envelope: {np.max(np.abs(body)):.5f}")
        
        # Envelope driven by transient
        excitation_env = np.zeros(total_samples)
        if transient_len <= total_samples:
            excitation_env[:transient_len] = np.abs(transient)  # Use transient energy
        else:
            excitation_env = np.abs(transient[:total_samples])
        
        # Decay envelope for body (exponential fade after transient)
        decay_samples = int(decay_rate * self.sample_rate)
        if transient_len + decay_samples < total_samples:
            decay_env = np.ones(total_samples)
            decay_env[transient_len:transient_len + decay_samples] = np.exp(-np.linspace(0, 5, decay_samples))  # Fast decay
            decay_env[transient_len + decay_samples:] = decay_env[transient_len + decay_samples - 1]  # Sustain tail
        else:
            decay_env = np.exp(-5 * t / length)  # Full exponential if transient dominates
        body_env = excitation_env + decay_env  # Combine excitation + decay
        body_env = body_env / np.max(body_env)  # Normalize envelope to 1.0
        body *= body_env
        print(f"Body max amp after envelope: {np.max(np.abs(body)):.5f}")
        
        # Output with transient infusion
        output = body
        if transient_len <= total_samples:
            output[:transient_len] += transient  # Add transient directly
        else:
            output += transient[:total_samples]
        print(f"Output max amp pre-normalize: {np.max(np.abs(output)):.5f}")
        
        # Normalize
        output = output / np.max(np.abs(output))
        print(f"Output final max amp: {np.max(np.abs(output)):.5f}")
        
        # Save
        sf.write(output_file, output, self.sample_rate, subtype='PCM_16')
        print(f"Sound saved to {output_file}")

# Test it
if __name__ == "__main__":
    synth = PhysicalModelingSynth()
    synth.synthesize(frequency=510, length=2.0, transient_file="input.transient", 
                     spectral_file="input.spectral", output_file="pm_synth.wav", decay_rate=0.5)