import numpy as np
import scipy.io.wavfile as wav
import os

def load_wav(file_path):
    """Load a WAV file and return sample rate and normalized audio data."""
    sample_rate, data = wav.read(file_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)  # Convert to mono if stereo
    data = data.astype(np.float32) / 32767  # Normalize to -1 to 1
    return sample_rate, data

def save_wav(file_path, sample_rate, data):
    """Save audio data as a WAV file."""
    data = np.int16(data * 32767)  # Convert back to 16-bit PCM
    wav.write(file_path, sample_rate, data)

def blend_audio(top_layer, bottom_layer, mode):
    """Blend two audio signals using various blend modes."""
    min_length = min(len(top_layer), len(bottom_layer))
    top_layer, bottom_layer = top_layer[:min_length], bottom_layer[:min_length]
    
    if mode == "add":
        result = top_layer + bottom_layer
    elif mode == "subtract":
        result = top_layer - bottom_layer
    elif mode == "multiply":
        result = top_layer * bottom_layer
    elif mode == "divide":
        result = np.where(bottom_layer != 0, top_layer / bottom_layer, 0)
    elif mode == "difference":
        result = np.abs(top_layer - bottom_layer)
    elif mode == "exclusion":
        result = top_layer + bottom_layer - 2 * top_layer * bottom_layer
    elif mode == "darken":
        result = np.minimum(top_layer, bottom_layer)
    elif mode == "lighten":
        result = np.maximum(top_layer, bottom_layer)
    elif mode == "color_dodge":
        result = np.where(bottom_layer != 1, top_layer / (1 - bottom_layer), 1)
    elif mode == "color_burn":
        result = np.where(top_layer != 0, 1 - (1 - bottom_layer) / top_layer, 0)
    else:
        raise ValueError(f"Unknown blend mode: {mode}")
    
    # Normalize result to prevent clipping
    result = np.clip(result, -1, 1)
    return result

def blend_wav_files(top_wav, bottom_wav, blend_mode, output_wav):
    """Blend two WAV files and save the output."""
    sr_top, top_layer = load_wav(top_wav)
    sr_bottom, bottom_layer = load_wav(bottom_wav)
    
    if sr_top != sr_bottom:
        raise ValueError("Sample rates do not match.")
    
    blended_audio = blend_audio(top_layer, bottom_layer, blend_mode)
    save_wav(output_wav, sr_top, blended_audio)
    print(f"Blended audio saved as {output_wav}")

# Example Usage
if __name__ == "__main__":
    blend_wav_files("t_layer.wav", "b_layer.wav", "color_burn", "blended_output.wav")
