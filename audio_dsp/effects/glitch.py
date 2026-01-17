import numpy as np
import soundfile as sf
from scipy.signal import resample, stft, istft
from audio_dsp.utils import load_audio, normalize_audio

def glitch_machine(input_file, output_file, n_segments=32, intensity=0.5, loop_length=2.0):
    """
    Glitch a WAV loop with weird effects.
    - input_file: Path to input WAV
    - output_file: Path to output WAV
    - n_segments: Number of segments (16, 32, 64, 128)
    - intensity: Fraction of segments to glitch (0.0–1.0)
    - loop_length: Duration of output loop in seconds
    """
    # Load WAV
    sr, audio = load_audio(input_file, mono=True)
    loop_samples = int(loop_length * sr)
    
    # Ensure audio fits loop length
    if len(audio) > loop_samples:
        audio = audio[:loop_samples]
    elif len(audio) < loop_samples:
        audio = np.tile(audio, (loop_samples // len(audio) + 1))[:loop_samples]
    
    # Split into segments
    segment_length = loop_samples // n_segments
    segments = [audio[i:i + segment_length] for i in range(0, loop_samples, segment_length)]
    if len(segments[-1]) < segment_length:
        segments[-1] = np.pad(segments[-1], (0, segment_length - len(segments[-1])), 'constant')
    
    # Determine how many segments to glitch
    n_glitch = int(n_segments * intensity)
    glitch_indices = np.random.choice(n_segments, n_glitch, replace=False)
    
    # Effects functions
    def retrigger(segment, divisions=4):
        chunk = segment[:len(segment) // divisions]
        return np.tile(chunk, divisions)[:len(segment)]
    
    def pitch_shift(segment, n_steps=4):
        # Simple pitch shift using resampling (also changes duration slightly)
        # n_steps: semitones to shift (positive = higher pitch)
        factor = 2 ** (n_steps / 12)
        # Resample to change pitch
        shifted = resample(segment, int(len(segment) / factor))
        # Resample back to original length
        return resample(shifted, len(segment))
    
    def reverse(segment):
        return segment[::-1]
    
    def quantize(segment, bits=8):
        levels = 2 ** bits
        return np.round(segment * (levels - 1)) / (levels - 1)
    
    def sample_rate_reduce(segment, factor=4):
        low_sr = sr // factor
        reduced = resample(segment, int(len(segment) * low_sr / sr))
        return resample(reduced, len(segment))
    
    def comb_delay(segment, delay_ms=10, feedback=0.5):
        delay_samples = int(sr * delay_ms / 1000)
        output = segment.copy()
        for i in range(delay_samples, len(segment)):
            output[i] += feedback * output[i - delay_samples]
        return np.clip(output, -1.0, 1.0)
    
    def time_stretch(segment, rate=2.0):
        # Simple time stretch using phase vocoder approach
        # rate > 1 = faster/shorter, rate < 1 = slower/longer
        n_fft = 512
        hop_length = n_fft // 4

        # STFT
        f, t, spec = stft(segment, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

        # Calculate new time axis
        new_length = int(spec.shape[1] / rate)
        if new_length < 2:
            new_length = 2

        # Interpolate magnitude and phase
        time_old = np.arange(spec.shape[1])
        time_new = np.linspace(0, spec.shape[1] - 1, new_length)

        mag = np.abs(spec)
        phase = np.angle(spec)

        # Interpolate each frequency bin
        new_mag = np.zeros((mag.shape[0], new_length), dtype=mag.dtype)
        new_phase = np.zeros((phase.shape[0], new_length), dtype=phase.dtype)

        for i in range(mag.shape[0]):
            new_mag[i] = np.interp(time_new, time_old, mag[i])
            new_phase[i] = np.interp(time_new, time_old, np.unwrap(phase[i]))

        new_spec = new_mag * np.exp(1j * new_phase)

        # ISTFT
        _, stretched = istft(new_spec, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

        # Adjust to original segment length
        if len(stretched) > len(segment):
            return stretched[:len(segment)]
        return np.pad(stretched, (0, len(segment) - len(stretched)), 'constant')
    
    def ring_mod(segment, freq=100):
        t = np.arange(len(segment)) / sr
        modulator = np.sin(2 * np.pi * freq * t)
        return segment * modulator
    
    def granular_chop(segment, grain_size=0.02):
        grain_samples = int(sr * grain_size)
        grains = [segment[i:i+grain_samples] for i in range(0, len(segment), grain_samples)]
        np.random.shuffle(grains)
        output = np.concatenate([g[:grain_samples] for g in grains if len(g) > 0])[:len(segment)]
        if len(output) < len(segment):
            output = np.pad(output, (0, len(segment) - len(output)), 'constant')
        return output
    
    def bit_flip(segment, flip_prob=0.05):
        # Simulate bit flipping by randomly inverting samples
        mask = np.random.random(len(segment)) < flip_prob
        return np.where(mask, -segment, segment)
    
    def spectral_freeze(segment):
        n_fft = 512
        hop_length = 256
        # STFT using scipy
        f, t, spec = stft(segment, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        # Freeze a random frame
        if spec.shape[1] > 0:
            freeze_frame = np.random.randint(spec.shape[1])
            frozen_spec = np.tile(spec[:, [freeze_frame]], (1, spec.shape[1]))
            # ISTFT
            _, output = istft(frozen_spec, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
            # Adjust length
            if len(output) > len(segment):
                return output[:len(segment)]
            return np.pad(output, (0, len(segment) - len(output)), 'constant')
        return segment
    
    # Available effects
    effects = [
        ("retrigger", lambda x: retrigger(x, divisions=4)),
        ("pitch_shift", lambda x: pitch_shift(x, n_steps=4)),
        ("reverse", reverse),
        ("quantize", lambda x: quantize(x, bits=6)),
        ("sample_rate_reduce", lambda x: sample_rate_reduce(x, factor=4)),
        ("comb_delay", lambda x: comb_delay(x, delay_ms=10, feedback=0.5)),
        ("time_stretch", lambda x: time_stretch(x, rate=2.0)),
        ("ring_mod", lambda x: ring_mod(x, freq=100)),
        ("granular_chop", lambda x: granular_chop(x, grain_size=0.02)),
        ("bit_flip", lambda x: bit_flip(x, flip_prob=0.05)),
        ("spectral_freeze", spectral_freeze)
    ]
    
    # Apply effects to selected segments
    for idx in glitch_indices:
        effect_name, effect_func = effects[np.random.randint(len(effects))]
        segments[idx] = effect_func(segments[idx])
        print(f"Applied {effect_name} to segment {idx}")
    
    # Reassemble audio
    output_audio = np.concatenate(segments)
    output_audio = normalize_audio(output_audio)
    
    # Save output
    sf.write(output_file, output_audio, sr, subtype='PCM_16')
    print(f"Glitched loop saved to {output_file}")

# Example usage
if __name__ == "__main__":
    input_file = "sequence.wav"  # Replace with your WAV file
    glitch_machine(
        input_file=input_file,
        output_file="glitched_loop_weird.wav",
        n_segments=16,
        intensity=1.0,
        loop_length=8.0
    )
