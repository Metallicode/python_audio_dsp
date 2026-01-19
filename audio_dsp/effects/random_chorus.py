import numpy as np
from scipy.interpolate import interp1d


def chorus(input_signal, sample_rate=44100, mix=0.5, amount=1.0, speed=0.1, n_clones=3, delay=0):
    """
    Apply a random chorus effect to an input signal.

    Args:
        input_signal: Input audio array (mono, normalized to ±1)
        sample_rate: Sample rate in Hz (default 44100)
        mix: Wet/dry mix 0–1 (default 0.5)
        amount: Pitch modulation amount (default 1.0)
        speed: Modulation speed (default 0.1)
        n_clones: Number of chorus voices (default 3)
        delay: Base delay in seconds (default 0)

    Returns:
        Output audio array with chorus effect applied
    """
    # Ensure input is float64 and normalized
    data = np.array(input_signal, dtype=np.float64)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    max_amp = np.max(np.abs(data))
    if max_amp > 0:
        data = data / max_amp

    length = len(data)
    output = np.copy(data)
    t = np.arange(length) / sample_rate

    # Process each clone
    for _ in range(n_clones):
        # Adjusted speed scaling for chorus-appropriate rates
        steps = max(10, int(length / (sample_rate * (1/speed) * 100)))  # Slower base rate
        random_walk = np.random.normal(0, 1, steps)
        random_walk = np.cumsum(random_walk)
        random_walk = random_walk / np.max(np.abs(random_walk)) * amount

        # Interpolate modulator to audio length
        modulator_times = np.linspace(0, t[-1], steps)
        modulator = interp1d(modulator_times, random_walk, kind='cubic', fill_value="extrapolate")(t)

        # Calculate time warping for pitch shift
        delay_samples = int(delay * sample_rate)
        time_warp = np.arange(length) + delay_samples + (modulator * 1000)

        # Ensure time_warp stays within bounds
        time_warp = np.clip(time_warp, 0, length - 1)

        # Interpolate the audio
        interpolator = interp1d(np.arange(length), data, kind='linear', fill_value=0, bounds_error=False)
        modulated = interpolator(time_warp)

        # Add to output
        output += modulated / n_clones

    # Mix wet and dry signals
    output = (1 - mix) * data + mix * output

    # Normalize output
    max_amp = np.max(np.abs(output))
    if max_amp > 0:
        output = output / max_amp

    return output


def random_chorus(input_file, output_file, mix=0.5, amount=1.0, speed=0.1, n_clones=3, delay=0):
    """
    Legacy file-based chorus function. Use chorus() for array-based processing.

    Args:
        input_file: Path to input WAV
        output_file: Path to output WAV
        mix: Wet/dry mix 0–1
        amount: Pitch modulation amount
        speed: Modulation speed
        n_clones: Number of chorus voices
        delay: Base delay in seconds
    """
    import soundfile as sf
    from audio_dsp.utils import wav_io as wavfile

    # Read the input WAV file
    sample_rate, data = wavfile.read(input_file)

    # Convert to float32 and normalize
    if data.dtype != np.float32 and data.dtype != np.float64:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max

    # Process using the new array-based function
    output = chorus(data, sample_rate, mix, amount, speed, n_clones, delay)

    # Write to WAV file
    sf.write(output_file, output, sample_rate, subtype='PCM_16')
    print(f"Processed {input_file} and saved to {output_file}")


# Example usage
if __name__ == "__main__":
    from audio_dsp.utils import wav_io as wavfile

    # Load sample data
    sample_rate, data = wavfile.read("input.wav")
    if data.dtype != np.float32 and data.dtype != np.float64:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Apply chorus effect
    chorused = chorus(data, sample_rate, mix=0.9, amount=5.0, speed=0.01, n_clones=3, delay=0.1)

    # Save output
    wavfile.write("output_chorus.wav", sample_rate, chorused.astype(np.float32))
    print("Chorus effect saved to output_chorus.wav")
