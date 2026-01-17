import numpy as np
import soundfile as sf
import os
import ast
import re
import json
import random
from audio_dsp.utils import load_audio, normalize_audio

def load_cluster_samples(cluster_file="cluster_mapping.json", samples_dir="samples"):
    """Load samples from clusters, returning audio data and paths."""
    with open(cluster_file, "r") as f:
        cluster_map = json.load(f)

    samples = {}
    for cluster_id, sample_paths in cluster_map.items():
        samples[cluster_id] = []
        for path in sample_paths:
            full_path = os.path.join(samples_dir, os.path.basename(path))
            sr, audio_data = load_audio(full_path)
            samples[cluster_id].append((audio_data, full_path))
    print(f"Loaded clusters: { {k: len(v) for k, v in samples.items()} }")
    return samples

def parse_pattern(pattern_file="pattern.txt"):
    """Parse pattern file for tracks, volumes, and modifiers."""
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
            modifiers.append(("", 1))
        parsed_patterns.append(pattern)
    
    return parsed_patterns, volumes, modifiers

def sequencer(pattern_file="pattern.txt", cluster_file="cluster_mapping.json", samples_dir="samples", 
              output_file="sequence.wav", bpm=120, output_length=16, random_seed=None):
    """Generate a sequence with random cluster-based sample selection."""
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    print(f"Random seed: {random_seed if random_seed is not None else 'None'}")

    # Load cluster samples
    cluster_samples = load_cluster_samples(cluster_file, samples_dir)
    if not cluster_samples:
        raise ValueError("No clusters found in cluster_mapping.json")

    # Parse pattern and volumes
    patterns, volumes, modifiers = parse_pattern(pattern_file)
    if len(patterns) > len(cluster_samples):
        raise ValueError(f"More patterns ({len(patterns)}) than clusters ({len(cluster_samples)})")
    if len(volumes) < len(patterns):
        raise ValueError(f"Volume list ({len(volumes)}) shorter than number of tracks ({len(patterns)})")

    # Map tracks to clusters
    cluster_ids = list(cluster_samples.keys())
    track_to_cluster = {i + 1: cluster_ids[i % len(cluster_ids)] for i in range(len(patterns))}
    print(f"Track to cluster mapping: {track_to_cluster}")

    # Calculate base timing
    sr = 44100
    base_step_duration = 60 / bpm / 4  # Quarter note
    base_step_samples = int(base_step_duration * sr)

    # Calculate per-track step durations
    step_samples_list = []
    for _, (op, factor) in zip(patterns, modifiers):
        if op == "*":
            step_duration = base_step_duration / factor
        elif op == "/":
            step_duration = base_step_duration * factor
        else:
            step_duration = base_step_duration
        step_samples_list.append(int(step_duration * sr))

    # Calculate total output length
    total_duration_seconds = output_length
    total_samples = int(total_duration_seconds * sr)
    print(f"Output length: {total_duration_seconds}s ({total_samples} samples)")

    # Extend patterns to fit total length
    extended_patterns = []
    for pattern, step_samples in zip(patterns, step_samples_list):
        pattern_steps = len(pattern)
        pattern_duration = pattern_steps * step_samples
        repeats = (total_samples // pattern_duration) + 1
        extended_pattern = (pattern * repeats)[:total_samples // step_samples + 1]
        extended_patterns.append(extended_pattern)

    # Log patterns
    print("Parsed patterns and modifiers:")
    for i, (orig, ext, (op, factor), step_samples) in enumerate(zip(patterns, extended_patterns, modifiers, step_samples_list), 1):
        mod_str = f"{op}{factor}" if op else "none"
        print(f"Track {i} (Cluster {track_to_cluster[i]}): {orig} → {ext} (modifier: {mod_str}, step_duration: {step_samples/sr:.5f}s)")

    # Process each track
    output = np.zeros(total_samples)
    for track_idx, (pattern, volume, step_samples) in enumerate(zip(extended_patterns, volumes, step_samples_list), 1):
        cluster_id = track_to_cluster[track_idx]
        cluster_sample_list = cluster_samples[cluster_id]
        if not cluster_sample_list:
            print(f"Warning: No samples in cluster {cluster_id} for track {track_idx}, skipping")
            continue

        print(f"Track {track_idx} (Cluster {cluster_id}) triggers:")
        for step, trigger in enumerate(pattern):
            if trigger == "1":
                # Randomly select a sample and its path
                sample_data, sample_path = random.choice(cluster_sample_list)
                sample = sample_data * volume
                sample_len = len(sample)
                start = step * step_samples
                if start < total_samples:
                    end = min(start + sample_len, total_samples)
                    sample_segment = sample[:end - start]
                    output[start:end] += sample_segment
                    print(f"Trigger at {start/sr:.5f}s with {os.path.basename(sample_path)}")

    # Normalize and save
    output = normalize_audio(output)
    sf.write(output_file, output, sr, subtype='PCM_16')
    print(f"Sequence saved to {output_file} (duration: {total_samples/sr:.5f}s)")

if __name__ == "__main__":
    sequencer(
        pattern_file="pattern.txt",
        cluster_file="cluster_mapping.json",
        samples_dir="cluster_samples",
        output_file="sequence.wav",
        bpm=180,
        output_length=64,
        random_seed=42
    )