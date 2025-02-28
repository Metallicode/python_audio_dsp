import numpy as np
import soundfile as sf
import json

class PhysicalModelingSynth:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def synthesize(self, frequency, length, transient_file, spectral_file, output_file):
        """
        Synthesize sound with crossfaded transient and spectral body.
        - frequency: Target frequency in Hz
        - length: Desired duration in seconds
        - transient_file: .transient file path
        - spectral_file: .spectral file path
        - output_file: Output WAV file path
        """
        # Load transient
        with open(transient_file, 'r') as f:
            transient_data = json.load(f)
        transient = np.array(transient_data["samples"])
        transient_len = len(transient)
        transient = transient / np.max(np.abs(transient))  # Normalize to 1.0
        print(f"Transient loaded: {transient_len} samples ({transient_len/self.sample_rate:.3f}s), max amp: {np.max(np.abs(transient)):.5f}")
        
        # Save transient for debugging
        sf.write("transient_debug.wav", transient, self.sample_rate, subtype='PCM_16')
        print("Saved transient_debug.wav")
        
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
        body = body / np.max(np.abs(body))  # Normalize to 1.0
        print(f"Body max amp: {np.max(np.abs(body)):.5f}")
        
        # Crossfade envelopes
        transient_env = np.ones(total_samples)
        body_env = np.ones(total_samples)
        if transient_len < total_samples:
            # Transient fades out over its length
            transient_env[:transient_len] = np.linspace(1, 0, transient_len)
            transient_env[transient_len:] = 0
            # Body fades in over transient length
            body_env[:transient_len] = np.linspace(0, 1, transient_len)
        else:
            # If transient is longer, truncate and adjust
            transient_env = np.linspace(1, 0, total_samples)
            body_env = np.linspace(0, 1, total_samples)
        print(f"Crossfade set over {min(transient_len, total_samples)} samples")
        
        # Apply envelopes
        transient_signal = np.zeros(total_samples)
        if transient_len <= total_samples:
            transient_signal[:transient_len] = transient
        else:
            transient_signal = transient[:total_samples]
        transient_signal *= transient_env
        print(f"Transient signal max amp: {np.max(np.abs(transient_signal)):.5f}")
        
        body_signal = body * body_env
        print(f"Body signal max amp: {np.max(np.abs(body_signal)):.5f}")
        
        # Combine
        output = transient_signal + body_signal
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
    synth.synthesize(frequency=310, length=2.0, transient_file="input.transient", 
                     spectral_file="input.spectral", output_file="pm_synth.wav")