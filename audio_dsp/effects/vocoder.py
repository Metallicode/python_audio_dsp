import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from audio_dsp.utils import load_audio, normalize_audio, resample_audio

def generate_carrier(sr, length, type="noise", freq=100):
    """Generate internal carrier if no WAV provided."""
    t = np.arange(length) / sr
    if type == "noise":
        noise = np.random.random(length) * 2 - 1
        sos = butter(4, [100, 5000], btype='band', fs=sr, output='sos')
        return sosfilt(sos, noise)
    elif type == "sawtooth":
        return 2 * (t * freq - np.floor(t * freq + 0.5))
    else:
        raise ValueError("Carrier type must be 'noise' or 'sawtooth'")


def _frame_audio(audio, frame_length, hop_length):
    """Frame audio into overlapping segments (replacement for librosa.util.frame)."""
    n_frames = (len(audio) - frame_length) // hop_length + 1
    frames = np.zeros((frame_length, n_frames))
    for i in range(n_frames):
        start = i * hop_length
        frames[:, i] = audio[start:start + frame_length]
    return frames


def vocoder(carrier, modulator, sr=None, n_filters=32, freq_range=(20, 20000),
            carrier_type="noise", carrier_freq=100, output_file=None):
    """
    Vocoder with band-pass filters, using shortest input length.

    Parameters:
    - carrier: Carrier signal as numpy array or path to WAV file.
               If None, generates carrier using carrier_type.
    - modulator: Modulator signal as numpy array or path to WAV file.
    - sr: Sample rate (required if inputs are arrays, ignored if loading from files)
    - n_filters: Number of band-pass filters
    - freq_range: Frequency range for filters (Hz)
    - carrier_type: 'noise' or 'sawtooth' if carrier is None
    - carrier_freq: Frequency for sawtooth carrier (Hz)
    - output_file: Path to output WAV (optional, if None returns array)

    Returns:
    - Vocoded audio as numpy array (if output_file is None)
    """
    # Load or use modulator
    sr, modulator = load_audio(modulator, sr=sr, mono=True)

    # Load, generate, or use carrier
    if carrier is None:
        carrier = generate_carrier(sr, len(modulator), type=carrier_type, freq=carrier_freq)
    elif isinstance(carrier, str):
        sr_carrier, carrier = load_audio(carrier, mono=True)
        if sr_carrier != sr:
            carrier = resample_audio(carrier, sr_carrier, sr)
    # else: carrier is already an array

    # Use shortest length
    min_length = min(len(modulator), len(carrier))
    modulator = modulator[:min_length]
    carrier = carrier[:min_length]

    # Design filter bank (log-spaced center frequencies)
    low_freq, high_freq = freq_range
    center_freqs = np.logspace(np.log10(low_freq), np.log10(high_freq), n_filters)
    bandwidth = (center_freqs[1:] - center_freqs[:-1]) / 2
    bandwidth = np.concatenate(([center_freqs[0]], bandwidth, [high_freq - center_freqs[-1]]))

    # Process each band
    output = np.zeros(min_length)
    frame_length = 1024
    hop_length = 256

    nyquist = sr / 2 - 1  # Stay safely below Nyquist
    for i in range(n_filters):
        # Band-pass filter design
        f_low = max(20, center_freqs[i] - bandwidth[i] / 2)
        f_high = min(nyquist, center_freqs[i] + bandwidth[i] / 2)
        if f_low >= f_high:
            continue  # Skip invalid filter bands
        sos = butter(4, [f_low, f_high], btype='band', fs=sr, output='sos')

        # Filter modulator and get envelope
        mod_band = sosfilt(sos, modulator)
        env_frames = np.abs(_frame_audio(mod_band, frame_length=frame_length, hop_length=hop_length))
        env = np.mean(env_frames, axis=0)

        # Resample envelope to match audio length
        env = resample_audio(env, sr / hop_length, sr)
        if len(env) > min_length:
            env = env[:min_length]
        elif len(env) < min_length:
            env = np.pad(env, (0, min_length - len(env)), 'edge')

        # Filter carrier and apply envelope
        car_band = sosfilt(sos, carrier)
        output += car_band * env

    # Normalize output
    output = normalize_audio(output)

    # Save to file or return array
    if output_file:
        sf.write(output_file, output, sr, subtype='PCM_16')
        print(f"Vocoded audio saved to {output_file}")
    return output, sr

# Example usage
if __name__ == "__main__":
    # With carrier and modulator WAV files
    output, sr = vocoder(
        carrier="synth.wav",
        modulator="voice.wav",
        output_file="vocoded_with_carrier.wav",
        n_filters=32,
        freq_range=(20, 20000)
    )

    # With generated noise carrier
    output, sr = vocoder(
        carrier=None,
        modulator="voice.wav",
        output_file="vocoded_noise.wav",
        n_filters=32,
        freq_range=(20, 20000),
        carrier_type="noise"
    )

    # With generated sawtooth carrier
    output, sr = vocoder(
        carrier=None,
        modulator="voice.wav",
        output_file="vocoded_sawtooth.wav",
        n_filters=64,
        freq_range=(20, 20000),
        carrier_type="sawtooth",
        carrier_freq=50
    )

    # Example with numpy arrays (no file output)
    # from audio_dsp.utils import load_audio
    # sr, carrier_array = load_audio("synth.wav")
    # sr, modulator_array = load_audio("voice.wav")
    # output, sr = vocoder(carrier_array, modulator_array, sr, n_filters=16)
