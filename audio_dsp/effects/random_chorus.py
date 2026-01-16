import numpy as np
from audio_dsp.utils import wav_io as wavfile
import soundfile as sf
from scipy.interpolate import interp1d

def random_chorus(input_file, output_file, mix=0.5, amount=1.0, speed=0.1, n_clones=3, delay=0):
    # Read the input WAV file
    sample_rate, data = wavfile.read(input_file)
    
    # Convert to float32 and normalize
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    
    # Handle stereo or mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)  # Convert stereo to mono
    
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
    output = output / np.max(np.abs(output))
    
    # Write to WAV file
    sf.write(output_file, output, sample_rate, subtype='PCM_16')

# Example usage
if __name__ == "__main__":
    input_file = "input.wav"
    output_file = "output_random_chorus.wav"
    
    random_chorus(
        input_file=input_file,
        output_file=output_file,
        mix=0.9,        # Mostly wet
        amount=5.0,     # Moderate pitch modulation
        speed=0.01,      # Now a reasonable chorus speed
        n_clones=3,     # Three voices
        delay=0.1      # Small delay
    )
    print(f"Processed {input_file} and saved to {output_file}")