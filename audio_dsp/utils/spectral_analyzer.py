import numpy as np
import json
from audio_dsp.utils import load_audio

class SpectralAnalyzer:
    def __init__(self, sample_rate=44100, fft_size=2048):
        self.sample_rate = sample_rate
        self.fft_size = fft_size

    def analyze(self, wav_file, output_file, num_peaks=10):
        """
        Analyze a WAV file and extract top spectral components.
        - wav_file: Input WAV file path
        - output_file: Output .spectral file path
        - num_peaks: Number of spectral peaks to extract
        """
        # Load WAV file
        sr, audio = load_audio(wav_file, mono=True)
        if len(audio) < self.fft_size:
            audio = np.pad(audio, (0, self.fft_size - len(audio)), 'constant')

        # FFT
        fft_data = np.fft.fft(audio[:self.fft_size])
        freqs = np.fft.fftfreq(len(fft_data), 1 / sr)
        mags = np.abs(fft_data)
        phases = np.angle(fft_data)

        # Get positive frequencies only (up to Nyquist)
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        mags = mags[pos_mask]
        phases = phases[pos_mask]

        # Find top peaks
        peak_indices = np.argsort(mags)[-num_peaks:][::-1]
        spectral_data = {
            "peaks": [
                {"frequency": float(freqs[idx]), "amplitude": float(mags[idx]), "phase": float(phases[idx])}
                for idx in peak_indices
            ],
            "sample_rate": sr,
            "fft_size": self.fft_size
        }

        # Save to .spectral file (JSON format)
        with open(output_file, 'w') as f:
            json.dump(spectral_data, f, indent=4)
        print(f"Spectral data saved to {output_file}")

# Test it
if __name__ == "__main__":
    analyzer = SpectralAnalyzer()
    analyzer.analyze("input.wav", "input.spectral", num_peaks=10)
