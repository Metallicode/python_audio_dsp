import numpy as np
import librosa
import soundfile as sf
import json

class TransientExtractor:
    def __init__(self, sample_rate=44100, window_size=256):
        self.sample_rate = sample_rate
        self.window_size = window_size

    def extract(self, wav_file, output_file, threshold=0.01, max_transient_len=0.5):
        """
        Extract transient from a WAV file.
        - wav_file: Input WAV file path
        - output_file: Output .transient file path
        - threshold: Energy threshold for transient detection (default 0.01)
        - max_transient_len: Max transient duration in seconds
        """
        # Load WAV file
        audio, sr = librosa.load(wav_file, sr=self.sample_rate, mono=True)
        print(f"Loaded {wav_file}: {len(audio)} samples, max amplitude: {np.max(np.abs(audio)):.5f}")
        
        if len(audio) < self.window_size:
            audio = np.pad(audio, (0, self.window_size - len(audio)), 'constant')
            print(f"Padded audio to {len(audio)} samples")
        
        # RMS energy over short windows
        rms = librosa.feature.rms(y=audio, frame_length=self.window_size, hop_length=self.window_size//2)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=self.window_size//2)
        print(f"RMS range: {np.min(rms):.5f} to {np.max(rms):.5f}")
        
        # Find transient start
        transient_frames = np.where(rms > threshold)[0]
        if len(transient_frames) > 0:
            transient_start_idx = transient_frames[0] * (self.window_size // 2)
        else:
            print(f"No transient found above threshold {threshold}. Using start of audio.")
            transient_start_idx = 0  # Fallback to first window
        
        transient_end_idx = min(transient_start_idx + int(max_transient_len * sr), len(audio))
        transient = audio[transient_start_idx:transient_end_idx]
        
        # Save to .transient file
        transient_data = {
            "samples": transient.tolist(),
            "sample_rate": sr,
            "length": len(transient) / sr
        }
        with open(output_file, 'w') as f:
            json.dump(transient_data, f, indent=4)
        print(f"Transient saved to {output_file} (start: {transient_start_idx/sr:.3f}s, length: {len(transient)/sr:.3f}s)")

# Test it
if __name__ == "__main__":
    extractor = TransientExtractor()
    extractor.extract("input.wav", "input.transient", threshold=0.01)