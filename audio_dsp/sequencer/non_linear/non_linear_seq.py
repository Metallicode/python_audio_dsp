import numpy as np
from audio_dsp.utils import wav_io as wavfile
from scipy.signal import resample
import matplotlib.pyplot as plt
import wave
import struct
import os
import glob
import random

def load_samples(samples_dir, target_rate=44100):
    """Load and resample WAV samples."""
    samples = {}
    for wav_file in glob.glob(os.path.join(samples_dir, "*.wav")):
        name = os.path.splitext(os.path.basename(wav_file))[0].lower()
        try:
            sample_rate, data = wavfile.read(wav_file)
            if len(data.shape) > 1:
                data = data[:, 0]
            if np.max(np.abs(data)) == 0:
                print(f"Warning: {name}.wav is silent or corrupted")
                continue
            if sample_rate != target_rate:
                num_samples = int(len(data) * target_rate / sample_rate)
                data = resample(data, num_samples)
                sample_rate = target_rate
            samples[name] = (sample_rate, data / np.max(np.abs(data)))
            print(f"Loaded {name}.wav: {len(data)} samples, rate {sample_rate}")
        except Exception as e:
            print(f"Error loading {wav_file}: {e}")
    if not samples:
        print(f"Error: No valid WAV files found in {samples_dir}")
    return samples

def parse_pattern(pattern_str):
    """Parse pattern string into events and loop count, supporting multiple '-'."""
    if '|' in pattern_str:
        pattern, loop_str = pattern_str.split('|')
        loops = int(loop_str)
    else:
        pattern, loops = pattern_str, 1
    
    events = []
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char in ['K', 'S', 'H', '.']:
            extend_count = 0
            # Count consecutive '-' characters
            j = i + 1
            while j < len(pattern) and pattern[j] == '-' and extend_count < 3:  # Cap at 3
                extend_count += 1
                j += 1
            events.append((char, extend_count))
            i = j
        else:
            print(f"Warning: Invalid character '{char}' at position {i}, skipping")
            i += 1
    return events, loops

def generate_liquid_timing(events, bpm=120, loop_length=4, sample_rate=44100):
    """Generate non-linear timings with scaled offsets for extend_count."""
    beat_duration = 60 / bpm
    loop_duration = loop_length * beat_duration
    samples_per_loop = int(loop_duration * sample_rate)
    
    grid_times = np.linspace(0, loop_duration, len(events), endpoint=False)
    
    liquid_times = []
    t = 0
    for i, (grid_time, (event, extend_count)) in enumerate(zip(grid_times, events)):
        A = (random.uniform(0.1, 0.6) + 0.1 * extend_count) * beat_duration
        omega = 2 * np.pi * 0.5
        gamma = 0.1
        phi = np.random.uniform(0, 2 * np.pi)
        offset = A * np.sin(omega * t + phi) * np.exp(-gamma * t)
        jitter = np.random.normal(0, 0.05 * beat_duration)
        swing = 0.1 * beat_duration if i % 2 == 1 else 0
        liquid_time = grid_time + offset + jitter + swing
        liquid_time = np.clip(liquid_time, 0, loop_duration - 0.001)
        liquid_times.append(liquid_time)
        t += beat_duration / len(events)
    
    liquid_times = np.sort(liquid_times)
    if len(liquid_times) > 0:
        liquid_times[-1] = loop_duration - 0.001
    print(f"Generated timings: {liquid_times} for {len(events)} events")
    return liquid_times, samples_per_loop

def generate_pattern(samples, events, liquid_times, loops, samples_per_loop, sample_rate=44100):
    """Generate audio with liquid rhythm."""
    total_samples = samples_per_loop * loops
    output = np.zeros(total_samples)
    
    sample_map = {'K': 'kick', 'S': 'snare', 'H': 'hihat'}
    
    if len(events) != len(liquid_times):
        print(f"Error: Mismatch between events ({len(events)}) and timings ({len(liquid_times)})")
        return output
    
    for loop in range(loops):
        loop_offset = loop * samples_per_loop
        for (event, extend_count), time in zip(events, liquid_times):
            if event in ['.']:
                continue
            sample_name = sample_map.get(event)
            if not sample_name or sample_name not in samples:
                print(f"Warning: Event '{event}' (expected {sample_name}) not found in samples: {list(samples.keys())}")
                continue
            sample_rate_s, sample_data = samples[sample_name]
            sample_pos = int(time * sample_rate)
            end_pos = sample_pos + loop_offset + len(sample_data)
            if end_pos <= total_samples:
                velocity = np.random.uniform(0.5, 1.0)
                output[sample_pos + loop_offset:end_pos] += sample_data * velocity
                print(f"Placed {sample_name} ({event}, extend={extend_count}) at sample {sample_pos + loop_offset}, loop {loop}")
            else:
                print(f"Skipped {sample_name} ({event}) at sample {sample_pos + loop_offset}: out of bounds (needs {len(sample_data)}, {total_samples - (sample_pos + loop_offset)} left)")
    
    return output

def visualize_pattern(events, liquid_times, loop_length, bpm):
    """Visualize the liquid rhythm with marker sizes for extend_count."""
    beat_duration = 60 / bpm
    loop_duration = loop_length * beat_duration
    
    plt.figure(figsize=(12, 4))
    for i in range(int(loop_length) + 1):
        plt.axvline(i * beat_duration, color='gray', linestyle='--', alpha=0.5)
    
    for i, ((event, extend_count), time) in enumerate(zip(events, liquid_times)):
        if event not in ['.']:
            # Larger marker for more '-' (100, 150, 200, 250)
            size = 100 + 50 * extend_count
            label = f"{event}{'-' * extend_count}" if i == 0 else ""
            plt.scatter(time, 1, s=size, marker='o', label=label, c='red')
        elif event == '.':
            plt.scatter(time, 0.5, s=50, c='blue', marker='x', label='Rest' if i == 0 else "")
    
    plt.title("Liquid Rhythm Pattern")
    plt.xlabel("Time (s)")
    plt.ylabel("Event")
    plt.ylim(0, 1.5)
    plt.legend()
    plt.savefig("liquid_rhythm_visualization.png")
    plt.close()

def main():
    samples_dir = "samples/"
    output_file = "liquid_rhythm.wav"
    pattern = "K.S-HK.S-H--.S.H--K---K-S.|4"  # Example with H--
    bpm = 160
    loop_length = 4
    
    print("Loading samples...")
    samples = load_samples(samples_dir)
    if not samples:
        print("Aborting: No samples loaded.")
        return
    
    print("Available samples:", list(samples.keys()))
    required_samples = {'K': 'kick', 'S': 'snare', 'H': 'hihat'}
    missing = [name for char, name in required_samples.items() if name not in samples]
    if missing:
        print(f"Error: Missing required samples: {missing}")
        return
    
    print("Generating pattern...")
    events, loops = parse_pattern(pattern)
    liquid_times, samples_per_loop = generate_liquid_timing(events, bpm, loop_length)
    output = generate_pattern(samples, events, liquid_times, loops, samples_per_loop)
    
    print("Saving visualization...")
    visualize_pattern(events, liquid_times, loop_length, bpm)
    
    print(f"Saving {output_file}...")
    if np.max(np.abs(output)) == 0:
        print("Error: Output is silent. Check sample placement or WAV files.")
        return
    output = output / (np.max(np.abs(output)) * 1.1)
    output = (output * 32767).astype(np.int16)
    with wave.open(output_file, 'w') as wav_file:
        wav_file.setparams((1, 2, 44100, len(output), 'NONE', 'not compressed'))
        for sample in output:
            wav_file.writeframes(struct.pack('h', sample))
    
    print(f"Created: {output_file}")
    print("Visualization saved as 'liquid_rhythm_visualization.png'")

if __name__ == "__main__":
    main()
