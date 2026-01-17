import numpy as np
import soundfile as sf
import os
import ast
import re
from audio_dsp.utils import load_audio, normalize_audio

def load_samples(samples_dir="samples"):
    samples = {}
    for filename in os.listdir(samples_dir):
        if filename.startswith(tuple(f"{i}_" for i in range(1, 10))) and filename.endswith(".wav"):
            track_num = int(filename.split("_")[0])
            sr, audio = load_audio(os.path.join(samples_dir, filename))
            samples[track_num] = audio
    return samples

def parse_pattern(pattern_file="pattern.txt"):
    with open(pattern_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines or not lines[-1].startswith("["):
        raise ValueError("Pattern file must end with a volume list like [1.0, 0.8, ...]")
    
    patterns = lines[:-1]
    volume_line = lines[-1]
    
    # Parse volumes
    try:
        volumes = ast.literal_eval(volume_line)
        if not isinstance(volumes, list) or not all(isinstance(v, (int, float)) for v in volumes):
            raise ValueError("Last line must be a valid list of numbers")
    except (ValueError, SyntaxError):
        raise ValueError("Invalid volume list format; use e.g., [1.0, 0.8, 0.5]")
    
    # Parse patterns and modifiers
    parsed_patterns = []
    modifiers = []
    for p in patterns:
        match = re.match(r"^(.*?)([*/]\d+)?$", p.replace(" ", ""))
        if not match:
            raise ValueError(f"Invalid pattern format: {p}")
        pattern, modifier = match.groups()
        if modifier:
            op = modifier[0]
            factor = int(modifier[1:])
            modifiers.append((op, factor))
        else:
            modifiers.append(("", 1))  # Default: no change
        parsed_patterns.append(pattern)
    
    return parsed_patterns, volumes, modifiers

def sequencer(pattern_file="pattern.txt", samples_dir="samples", output_file="sequence.wav", bpm=120):
    # Load samples
    samples = load_samples(samples_dir)
    if not samples:
        raise ValueError("No valid samples found in samples/ dir")
    
    # Parse pattern and volumes
    patterns, volumes, modifiers = parse_pattern(pattern_file)
    if len(patterns) > len(samples):
        raise ValueError(f"More patterns ({len(patterns)}) than samples ({len(samples)})")
    if len(volumes) < len(patterns):
        raise ValueError(f"Volume list ({len(volumes)}) shorter than number of tracks ({len(patterns)})")
    
    # Calculate base timing
    sr = 44100
    base_step_duration = 60 / bpm / 4  # 0.125s at 120 BPM
    base_step_samples = int(base_step_duration * sr)
    
    # Calculate per-track step durations and base track lengths
    track_durations = []
    step_samples_list = []
    for pattern, (op, factor) in zip(patterns, modifiers):
        if op == "*":
            step_duration = base_step_duration / factor  # Faster
        elif op == "/":
            step_duration = base_step_duration * factor  # Slower
        else:
            step_duration = base_step_duration  # Normal
        step_samples = int(step_duration * sr)
        step_samples_list.append(step_samples)
        track_base_duration = len(pattern) * step_samples
        track_durations.append(track_base_duration)
    
    # Find longest track duration and extend patterns
    max_duration = max(track_durations)  # In samples
    extended_patterns = []
    for pattern, step_samples in zip(patterns, step_samples_list):
        pattern_steps = len(pattern)
        pattern_duration = pattern_steps * step_samples
        repeats = (max_duration // pattern_duration) + 1
        extended_pattern = (pattern * repeats)[:max_duration // step_samples + 1]
        extended_patterns.append(extended_pattern)
    
    # Log for verification
    print("Parsed patterns and modifiers:")
    for i, (orig, ext, (op, factor), step_samples) in enumerate(zip(patterns, extended_patterns, modifiers, step_samples_list), 1):
        mod_str = f"{op}{factor}" if op else "none"
        print(f"Track {i}: {orig} → {ext} (modifier: {mod_str}, step_duration: {step_samples/sr:.5f}s)")
    
    # Calculate total samples
    total_samples = max_duration
    
    # Process each track
    output = np.zeros(total_samples)
    for track_idx, (pattern, volume, step_samples) in enumerate(zip(extended_patterns, volumes, step_samples_list), start=1):
        if track_idx not in samples:
            print(f"Warning: No sample for track {track_idx}, skipping")
            continue
        
        sample = samples[track_idx] * volume
        sample_len = len(sample)
        
        print(f"Track {track_idx} triggers:")
        for step, trigger in enumerate(pattern):
            if trigger == "1":
                start = step * step_samples
                if start < total_samples:  # Only add if within bounds
                    end = min(start + sample_len, total_samples)
                    sample_segment = sample[:end - start]
                    output[start:end] += sample_segment
                    print(f"Trigger at {start/sr:.5f}")
    
    # Normalize and save
    output = normalize_audio(output)
    sf.write(output_file, output, sr, subtype='PCM_16')
    print(f"Sequence saved to {output_file} (duration: {total_samples/sr:.5f}s)")

if __name__ == "__main__":
    sequencer(
        pattern_file="pattern.txt",
        samples_dir="samples",
        output_file="sequence.wav",
        bpm=120
    )