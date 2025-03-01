import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt

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

def vocoder(modulator_file, carrier_file=None, output_file="vocoded.wav", 
            n_filters=32, freq_range=(20, 20000), carrier_type="noise", carrier_freq=100):
    """
    Vocoder with band-pass filters, using shortest input length.
    - modulator_file: Path to modulator WAV
    - carrier_file: Path to carrier WAV (optional)
    - output_file: Path to output WAV
    - n_filters: Number of band-pass filters
    - freq_range: Frequency range for filters (Hz)
    - carrier_type: 'noise' or 'sawtooth' if no carrier_file
    - carrier_freq: Frequency for sawtooth carrier (Hz)
    """
    # Load modulator
    modulator, sr = librosa.load(modulator_file, sr=None, mono=True)
    
    # Load or generate carrier
    if carrier_file:
        carrier, sr_carrier = librosa.load(carrier_file, sr=None, mono=True)
        if sr_carrier != sr:
            carrier = librosa.resample(carrier, orig_sr=sr_carrier, target_sr=sr)
    else:
        carrier = generate_carrier(sr, len(modulator), type=carrier_type, freq=carrier_freq)
    
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
    n_frames = (min_length - frame_length) // hop_length + 1
    
    for i in range(n_filters):
        # Band-pass filter design
        f_low = max(20, center_freqs[i] - bandwidth[i] / 2)
        f_high = min(sr / 2, center_freqs[i] + bandwidth[i] / 2)
        sos = butter(4, [f_low, f_high], btype='band', fs=sr, output='sos')
        
        # Filter modulator and get envelope
        mod_band = sosfilt(sos, modulator)
        env_frames = np.abs(librosa.util.frame(mod_band, frame_length=frame_length, hop_length=hop_length))
        env = np.mean(env_frames, axis=0)
        
        # Resample envelope to match audio length (using default scipy method if samplerate unavailable)
        env = librosa.resample(env, orig_sr=sr / hop_length, target_sr=sr, res_type='kaiser_best')
        if len(env) > min_length:
            env = env[:min_length]
        elif len(env) < min_length:
            env = np.pad(env, (0, min_length - len(env)), 'edge')
        
        # Filter carrier and apply envelope
        car_band = sosfilt(sos, carrier)
        output += car_band * env
    
    # Normalize and save
    output = librosa.util.normalize(output)
    sf.write(output_file, output, sr, subtype='PCM_16')
    print(f"Vocoded audio saved to {output_file}")

# Example usage
if __name__ == "__main__":
    vocoder(
        modulator_file="voice.wav",
        carrier_file="synth.wav",
        output_file="vocoded_with_carrier.wav",
        n_filters=32,
        freq_range=(20, 20000)
    )
    
    vocoder(
        modulator_file="voice.wav",
        carrier_file=None,
        output_file="vocoded_noise.wav",
        n_filters=32,
        freq_range=(20, 20000),
        carrier_type="noise"
    )
    
    vocoder(
        modulator_file="voice.wav",
        carrier_file=None,
        output_file="vocoded_sawtooth.wav",
        n_filters=64,
        freq_range=(20, 20000),
        carrier_type="sawtooth",
        carrier_freq=50
    )