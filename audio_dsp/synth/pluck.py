# import numpy as np
# from audio_dsp.utils import wav_io as wavfile

# SAMPLE_RATE = 44100

# def save_wav(file_path, data):
#     data = np.clip(data, -1, 1)
#     wavfile.write(file_path, SAMPLE_RATE, (data * 32767).astype(np.int16))
#     print(f"Saved to {file_path}")

# def karplus_strong(length, frequency, intensity, decay=0.996):
#     """Generate a plucked string sound using Karplus-Strong synthesis."""
#     # Calculate buffer size (delay line length) based on frequency
#     period = int(SAMPLE_RATE / frequency)  # Samples per cycle
#     total_samples = int(length * SAMPLE_RATE)
#     output = np.zeros(total_samples, dtype=np.float32)
    
#     # Initialize buffer with noise (the pluck)
#     buffer = np.random.uniform(-1, 1, period) * intensity
    
#     # Fill initial output with the pluck
#     if period > total_samples:
#         output[:total_samples] = buffer[:total_samples]
#         return output
#     else:
#         output[:period] = buffer
    
#     # Karplus-Strong loop: feedback with damping
#     for i in range(period, total_samples):
#         # Average previous two samples (low-pass filter)
#         output[i] = decay * 0.5 * (output[i - period] + output[i - period - 1])
    
#     return output

# def generate_string_pluck(output_file, length=2.0, frequency=440.0, intensity=1.0):
#     """Generate and save a string pluck sound."""
#     print(f"Generating string pluck: {length}s, {frequency}Hz, intensity {intensity}...")
#     signal = karplus_strong(length, frequency, intensity)
    
#     print(f"Saving to {output_file}...")
#     save_wav(output_file, signal)

# if __name__ == "__main__":
#     # Example parameters
#     output_file = "string_pluck.wav"
#     length = 2.0      # Duration in seconds
#     frequency = 50.0 # Hz (A4)
#     intensity = 0.8   # Amplitude scale (0 to 1)
    
#     generate_string_pluck(output_file, length, frequency, intensity)

import numpy as np
from audio_dsp.utils import wav_io as wavfile

SAMPLE_RATE = 44100

def save_wav(file_path, data):
    data = np.clip(data, -1, 1)
    wavfile.write(file_path, SAMPLE_RATE, (data * 32767).astype(np.int16))
    print(f"Saved to {file_path}")

def karplus_strong(length, frequency, intensity, damping=0.5, brightness=0.5, 
                   mute_strength=0.5, pitch_bend=0.5, pluck_position=0.5, plectrum_size=0.5):
    """Generate a plucked string sound with plectrum size control."""
    # Calculate buffer size and total samples
    period = int(SAMPLE_RATE / frequency)
    total_samples = int(length * SAMPLE_RATE)
    output = np.zeros(total_samples, dtype=np.float32)
    
    # Plectrum size affects noise duration and amplitude
    pluck_duration = int(period * (0.5 + plectrum_size * 0.5))  # 50% to 100% of period
    pluck_duration = min(pluck_duration, total_samples)  # Cap at signal length
    pluck_amplitude = intensity * (0.5 + plectrum_size * 0.5)  # 50% to 100% of intensity
    
    # Initialize buffer with noise, shaped by pluck position and plectrum size
    buffer = np.random.uniform(-1, 1, period) * pluck_amplitude
    if pluck_position < 1.0:
        env = np.sin(np.pi * np.linspace(0, 1, period))  # Half-sine shape
        env_weight = pluck_position
        buffer = buffer * (1 - env_weight) + buffer * env * env_weight
    
    # Apply plectrum duration
    if pluck_duration < period:
        buffer[pluck_duration:] = 0  # Zero out beyond plectrum duration
    
    if period > total_samples:
        output[:total_samples] = buffer[:total_samples]
        return output
    else:
        output[:period] = buffer
    
    # Map parameters
    decay = 0.99 + (1 - damping) * 0.009
    filter_weight = brightness * (1 - pluck_position * 0.5)
    
    # Pitch bend
    t = np.linspace(0, length, total_samples, endpoint=False)
    bend_amount = pitch_bend * 0.5
    pitch_factor = 2 ** (bend_amount / 12)
    freq_mod = frequency * (1 + (pitch_factor - 1) * np.exp(-5 * t))
    
    # Mute envelope
    mute_factor = mute_strength * np.linspace(0, 1, total_samples)
    
    # Karplus-Strong loop
    for i in range(period, total_samples):
        prev_sample = output[i - period]
        prev_prev_sample = output[i - period - 1]
        filtered = filter_weight * prev_sample + (1 - filter_weight) * prev_prev_sample
        mute_weight = 1 - mute_factor[i]
        filtered = mute_weight * filtered + (1 - mute_weight) * 0.5 * (prev_sample + prev_prev_sample)
        output[i] = decay * filtered
    
    # Apply pitch modulation
    phase = 2 * np.pi * freq_mod * t
    output = output * np.sin(phase)
    
    return output

def generate_string_pluck(output_file, length=2.0, frequency=440.0, intensity=1.0, 
                         damping=0.5, brightness=0.5, mute_strength=0.5, 
                         pitch_bend=0.5, pluck_position=0.5, plectrum_size=0.5):
    """Generate and save a string pluck sound with all parameters."""
    print(f"Generating string pluck: {length}s, {frequency}Hz, intensity {intensity}, "
          f"damping {damping}, brightness {brightness}, mute_strength {mute_strength}, "
          f"pitch_bend {pitch_bend}, pluck_position {pluck_position}, "
          f"plectrum_size {plectrum_size}...")
    signal = karplus_strong(length, frequency, intensity, damping, brightness, 
                            mute_strength, pitch_bend, pluck_position, plectrum_size)
    
    print(f"Saving to {output_file}...")
    save_wav(output_file, signal)

if __name__ == "__main__":
    output_file = "string_pluck.wav"
    length = 2.0
    frequency = 90.0
    intensity = 0.6
    damping = 0.8
    brightness = 0.5
    mute_strength = 0.5
    pitch_bend = 0.5
    pluck_position = 0.9
    plectrum_size = 0.8  # 0 (small/thin), 1 (large/thick)
    
    generate_string_pluck(output_file, length, frequency, intensity, damping, brightness, 
                          mute_strength, pitch_bend, pluck_position, plectrum_size)