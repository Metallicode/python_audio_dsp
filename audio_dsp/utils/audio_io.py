"""
Audio I/O utilities for consistent audio loading and saving across the package.

Uses scipy.io.wavfile as the lightweight backend for WAV file operations.
"""

import numpy as np
from audio_dsp.utils import wav_io as wavfile
from scipy.signal import resample_poly
from math import gcd


def load_audio(audio_input, sr=None, mono=True):
    """
    Load audio from a file path or pass through a numpy array.

    Parameters:
    - audio_input: File path (str) or numpy array
    - sr: Sample rate (required if audio_input is an array, ignored if loading from file)
    - mono: Convert to mono if True (default True)

    Returns:
    - sr: Sample rate
    - audio: Audio data as float32 numpy array normalized to [-1, 1]

    Note: Return order matches wavfile.read() convention: (sample_rate, data)

    Raises:
    - ValueError: If audio_input is an array and sr is not provided
    - TypeError: If audio_input is neither a string nor a numpy array
    """
    if isinstance(audio_input, str):
        # Load from file
        file_sr, data = wavfile.read(audio_input)

        # Convert to float32 normalized to [-1, 1]
        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.float32:
            audio = data
        elif data.dtype == np.float64:
            audio = data.astype(np.float32)
        elif data.dtype == np.uint8:
            audio = (data.astype(np.float32) - 128) / 128.0
        else:
            # Fallback for other dtypes
            audio = data.astype(np.float32)
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

        # Convert to mono if needed
        if mono and audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        return file_sr, audio

    elif isinstance(audio_input, np.ndarray):
        if sr is None:
            raise ValueError("sr (sample rate) is required when audio_input is an array")

        audio = audio_input.astype(np.float32)

        # Convert to mono if needed
        if mono and audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        return sr, audio

    else:
        raise TypeError(f"audio_input must be a file path (str) or numpy array, got {type(audio_input)}")


def save_audio(file_path, audio, sr, bit_depth=16):
    """
    Save audio to a WAV file.

    Parameters:
    - file_path: Output file path
    - audio: Audio data as numpy array (should be normalized to [-1, 1])
    - sr: Sample rate
    - bit_depth: Bit depth (16 or 24, default 16)
    """
    audio = np.clip(audio, -1.0, 1.0)

    if bit_depth == 16:
        audio_int = (audio * 32767).astype(np.int16)
    elif bit_depth == 24:
        # scipy.io.wavfile doesn't support 24-bit, use 32-bit float instead
        audio_int = audio.astype(np.float32)
    else:
        audio_int = (audio * 32767).astype(np.int16)

    wavfile.write(file_path, sr, audio_int)


def normalize_audio(audio, peak=1.0):
    """
    Normalize audio to a target peak amplitude.

    Parameters:
    - audio: Audio data as numpy array
    - peak: Target peak amplitude (default 1.0)

    Returns:
    - Normalized audio array
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio * (peak / max_val)
    return audio


def resample_audio(audio, orig_sr, target_sr):
    """
    Resample audio from one sample rate to another.

    Parameters:
    - audio: Audio data as numpy array
    - orig_sr: Original sample rate
    - target_sr: Target sample rate

    Returns:
    - Resampled audio array
    """
    if orig_sr == target_sr:
        return audio

    # Find GCD to simplify the ratio
    g = gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // g
    down = int(orig_sr) // g

    return resample_poly(audio, up, down).astype(np.float32)
