# import numpy as np
# import soundfile as sf

# class SuperCleanCompressor:
#     def __init__(self, sample_rate=44100):
#         self.sample_rate = sample_rate

#     def dB_to_linear(self, dB):
#         """Convert dB to linear gain."""
#         return 10 ** (dB / 20)

#     def linear_to_dB(self, linear):
#         """Convert linear gain to dB."""
#         return 20 * np.log10(np.maximum(linear, 1e-10))

#     def compress(self, input_file, output_file, mode="transparent", input_gain=0.0, threshold=-20.0, 
#                  ratio=4.0, attack=0.01, release=0.1, output_gain=0.0, limit=False):
#         """
#         Compress audio with vintage or transparent mode.
#         - input_file: Input WAV file path
#         - output_file: Output WAV file path
#         - mode: 'vintage' or 'transparent'
#         - input_gain: Input boost in dB
#         - threshold: Compression threshold in dB
#         - ratio: Compression ratio (e.g., 4.0 = 4:1)
#         - attack: Attack time in seconds
#         - release: Release time in seconds
#         - output_gain: Output gain in dB
#         - limit: True to hard-limit at 0 dBFS
#         """
#         # Load audio
#         audio, sr = sf.read(input_file)
#         if sr != self.sample_rate:
#             audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
#         if audio.ndim > 1:
#             audio = np.mean(audio, axis=1)  # Mono
        
#         # Apply input gain
#         input_gain_linear = self.dB_to_linear(input_gain)
#         audio = audio * input_gain_linear
#         print(f"Input max amp after gain: {np.max(np.abs(audio)):.5f}")
        
#         # Convert threshold to linear
#         threshold_linear = self.dB_to_linear(threshold)
        
#         # Level detection
#         if mode == "vintage":
#             rms_window = int(0.01 * self.sample_rate)
#             rms = np.sqrt(np.convolve(audio**2, np.ones(rms_window)/rms_window, mode='same'))
#             level = rms
#         else:
#             level = np.abs(audio)
#         print(f"Level max: {np.max(level):.5f}, threshold: {threshold_linear:.5f}")
        
#         # Gain reduction in dB
#         gain_reduction_db = np.zeros_like(level)
#         for i in range(len(level)):
#             if level[i] > threshold_linear:
#                 level_db = self.linear_to_dB(level[i])
#                 thresh_db = self.linear_to_dB(threshold_linear)
#                 excess_db = level_db - thresh_db
#                 gain_reduction_db[i] = -excess_db * (1 - 1/ratio)  # Correct GR formula
        
#         print(f"Raw GR max: {np.min(gain_reduction_db):.5f} dB")
        
#         # Smooth gain reduction
#         attack_samples = int(attack * self.sample_rate)
#         release_samples = int(release * self.sample_rate)
#         smoothed_gr_db = np.zeros_like(gain_reduction_db)
#         current_gr = 0.0
#         for i in range(len(gain_reduction_db)):
#             target_gr = gain_reduction_db[i]
#             if target_gr < current_gr:  # Attack (more reduction)
#                 alpha = np.exp(-1.0 / attack_samples)
#             else:  # Release (less reduction)
#                 alpha = np.exp(-1.0 / release_samples)
#             current_gr = alpha * current_gr + (1 - alpha) * target_gr
#             smoothed_gr_db[i] = current_gr
        
#         print(f"Smoothed GR max: {np.min(smoothed_gr_db):.5f} dB")
        
#         # Apply gain reduction
#         gain_linear = self.dB_to_linear(smoothed_gr_db)
#         output = audio * gain_linear
#         print(f"Output max amp pre-gain: {np.max(np.abs(output)):.5f}")
        
#         # Vintage saturation
#         if mode == "vintage":
#             output = np.tanh(output * 1.2) / 1.2
        
#         # Output gain
#         output_gain_linear = self.dB_to_linear(output_gain)
#         output *= output_gain_linear
#         print(f"Output max amp pre-limit: {np.max(np.abs(output)):.5f}")
        
#         # Limiter
#         if limit:
#             output = np.clip(output, -1.0, 1.0)
#             print(f"Output max amp post-limit: {np.max(np.abs(output)):.5f}")
        
#         # Normalize if needed
#         if not limit and np.max(np.abs(output)) > 1.0:
#             output = output / np.max(np.abs(output))
#             print(f"Output max amp post-normalize: {np.max(np.abs(output)):.5f}")
        
#         # Save
#         sf.write(output_file, output, self.sample_rate, subtype='PCM_16')
#         print(f"Compressed audio saved to {output_file}")

# # Test it
# if __name__ == "__main__":
#     compressor = SuperCleanCompressor()
    
#     # Extreme test: Should squash hard
#     compressor.compress("input.wav", "compressed_test.wav", mode="transparent", 
#                         input_gain=6.0, threshold=-30.0, ratio=3.0, attack=0.001, 
#                         release=0.01, output_gain=5.0, limit=False)


import librosa
import numpy as np
import soundfile as sf

class SuperCleanCompressor:
    def __init__(self, sample_rate=44100, oversample_factor=2):
        self.sample_rate = sample_rate
        self.oversample_factor = oversample_factor
        self.effective_sr = sample_rate * oversample_factor

    def dB_to_linear(self, dB):
        return 10 ** (dB / 20)

    def linear_to_dB(self, linear):
        return 20 * np.log10(np.maximum(linear, 1e-10))

    def compress(self, input_file, output_file, mode="transparent", input_gain=0.0, threshold=-20.0, 
                 ratio=4.0, attack=0.01, release=0.1, knee_width=6.0, output_gain=0.0, limit=False):
        """
        State-of-the-art compression with vintage or transparent mode.
        - input_file: Input WAV file path
        - output_file: Output WAV file path
        - mode: 'vintage' or 'transparent'
        - input_gain: Input boost in dB
        - threshold: Compression threshold in dB
        - ratio: Compression ratio (e.g., 4.0 = 4:1)
        - attack: Attack time in seconds
        - release: Base release time in seconds (adaptive in vintage)
        - knee_width: Knee transition width in dB (e.g., 6.0 = ±3 dB)
        - output_gain: Output gain in dB
        - limit: True to hard-limit at 0 dBFS
        """
        # Load audio
        audio, sr = sf.read(input_file)
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Oversample
        audio = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=self.effective_sr)
        total_samples = len(audio)
        
        # Apply input gain
        input_gain_linear = self.dB_to_linear(input_gain)
        audio = audio * input_gain_linear
        print(f"Input max amp after gain: {np.max(np.abs(audio)):.5f}")
        
        # Level detection
        if mode == "vintage":
            rms_window = int(0.01 * self.effective_sr)
            rms = np.sqrt(np.convolve(audio**2, np.ones(rms_window)/rms_window, mode='same'))
            level = rms
        else:
            level = np.abs(audio)
        print(f"Level max: {np.max(level):.5f}")
        
        # Gain reduction in dB
        threshold_linear = self.dB_to_linear(threshold)
        gain_reduction_db = np.zeros_like(level)
        for i in range(len(level)):
            level_db = self.linear_to_dB(level[i])
            thresh_db = self.linear_to_dB(threshold_linear)
            if level_db > thresh_db - knee_width/2:
                excess_db = level_db - thresh_db + knee_width/2
                if excess_db > knee_width:  # Above knee
                    gain_reduction_db[i] = -excess_db * (1 - 1/ratio)
                elif excess_db > 0:  # In knee
                    gain_reduction_db[i] = -0.5 * (excess_db**2) / (knee_width * (1 - 1/ratio))
        
        print(f"Raw GR max: {np.min(gain_reduction_db):.5f} dB")
        
        # Smooth gain reduction with adaptive release
        attack_samples = int(attack * self.effective_sr)
        release_samples = int(release * self.effective_sr)
        smoothed_gr_db = np.zeros_like(gain_reduction_db)
        current_gr = 0.0
        for i in range(len(gain_reduction_db)):
            target_gr = gain_reduction_db[i]
            if target_gr < current_gr:  # Attack
                alpha = np.exp(-1.0 / attack_samples)
            else:  # Release (adaptive in vintage)
                if mode == "vintage" and i > 0:
                    delta = abs(level[i] - level[i-1])
                    adaptive_release = release_samples * (1 + delta * 10)  # Faster on peaks
                    alpha = np.exp(-1.0 / min(adaptive_release, release_samples * 5))
                else:
                    alpha = np.exp(-1.0 / release_samples)
            current_gr = alpha * current_gr + (1 - alpha) * target_gr
            smoothed_gr_db[i] = current_gr
        
        print(f"Smoothed GR max: {np.min(smoothed_gr_db):.5f} dB")
        
        # Apply gain reduction
        gain_linear = self.dB_to_linear(smoothed_gr_db)
        output = audio * gain_linear
        print(f"Output max amp pre-gain: {np.max(np.abs(output)):.5f}")
        
        # Vintage saturation (asymmetric)
        if mode == "vintage":
            output = np.tanh(output * 1.5 + 0.1 * output**3) / 1.5  # Add cubic warmth
        
        # Output gain
        output_gain_linear = self.dB_to_linear(output_gain)
        output *= output_gain_linear
        print(f"Output max amp pre-limit: {np.max(np.abs(output)):.5f}")
        
        # Limiter
        if limit:
            output = np.clip(output, -1.0, 1.0)
            print(f"Output max amp post-limit: {np.max(np.abs(output)):.5f}")
        
        # Downsample
        output = librosa.resample(output, orig_sr=self.effective_sr, target_sr=self.sample_rate)
        
        # Normalize if needed
        if not limit and np.max(np.abs(output)) > 1.0:
            output = output / np.max(np.abs(output))
            print(f"Output max amp post-normalize: {np.max(np.abs(output)):.5f}")
        
        # Save
        sf.write(output_file, output, self.sample_rate, subtype='PCM_16')
        print(f"Compressed audio saved to {output_file}")

# Test it
if __name__ == "__main__":
    compressor = SuperCleanCompressor()
    compressor.compress("input.wav", "compressed_test.wav", mode="transparent", 
                        input_gain=10.0, threshold=-30.0, ratio=3.0, attack=0.001, 
                        release=0.01, knee_width=6.0, output_gain=5.0, limit=False)