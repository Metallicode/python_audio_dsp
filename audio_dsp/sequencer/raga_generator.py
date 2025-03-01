import numpy as np
import simpleaudio as sa
from datetime import datetime, timedelta
import time
import random

# Define ragas and their intervals
ragas_by_time = {
    "early_morning": {
        "time_range": (4, 7),
        "ragas": [
            {"name": "Bhairav", "intervals": [0, 1, 3, 4, 7, 8, 10, 12]},
            {"name": "Ramkali", "intervals": [0, 1, 3, 4, 7, 8, 10, 12]}
        ]
    },
    "late_morning": {
        "time_range": (7, 10),
        "ragas": [
            {"name": "Bilawal", "intervals": [0, 2, 4, 5, 7, 9, 11, 12]},
            {"name": "Jaunpuri", "intervals": [0, 2, 3, 5, 7, 8, 10, 12]}
        ]
    },
    "afternoon": {
        "time_range": (10, 14),
        "ragas": [
            {"name": "Sarang", "intervals": [0, 2, 4, 7, 9, 12]},
            {"name": "Brindavani Sarang", "intervals": [0, 2, 4, 7, 9, 12]}
        ]
    },
    "evening": {
        "time_range": (17, 22),
        "ragas": [
            {"name": "Yaman", "intervals": [0, 2, 4, 6, 7, 9, 11, 12]},
            {"name": "Kafi", "intervals": [0, 2, 3, 5, 7, 8, 10, 12]}
        ]
    },
    "night": {
        "time_range": (22, 4),
        "ragas": [
            {"name": "Malkauns", "intervals": [0, 3, 5, 8, 10, 12]},
            {"name": "Bageshree", "intervals": [0, 3, 5, 7, 8, 10, 12]}
        ]
    }
}

# Choose a raga based on the current time
def choose_raga():
    current_hour = datetime.now().hour
    for period, details in ragas_by_time.items():
        start, end = details["time_range"]
        if start < end:
            if start <= current_hour < end:
                return np.random.choice(details["ragas"])
        else:
            if current_hour >= start or current_hour < end:
                return np.random.choice(details["ragas"])
    return None

# Generate fractal rhythm with randomness
def generate_fractal_rhythm(core_pattern, depth=3, randomness=0.2):
    pattern = core_pattern
    for _ in range(depth):
        next_pattern = []
        for step in pattern:
            # Introduce randomness with a 20% chance
            if random.random() < randomness:
                step = 1 if step == 0 else 0
            next_pattern.append(step)
        pattern = pattern + next_pattern
    return pattern

# Generate a new random core pattern
def generate_random_core_pattern(length=6):
    return [random.choice([0, 1]) for _ in range(length)]

# Stochastic note selection
def select_next_note(current_note, intervals):
    distances = np.abs(np.array(intervals) - current_note)
    probabilities = np.exp(-distances)
    probabilities /= probabilities.sum()
    next_note = np.random.choice(intervals, p=probabilities)
    return next_note

# Generate sine wave with Attack/Decay envelope
def generate_sine_wave_with_envelope(frequency, duration, sample_rate=44100, attack=0.01, decay=0.1):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = np.sin(2 * np.pi * frequency * t)

    # Create envelope
    envelope = np.ones_like(waveform)
    attack_samples = int(sample_rate * attack)
    decay_samples = int(sample_rate * decay)
    sustain_samples = len(waveform) - attack_samples - decay_samples

    # Attack phase
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

    # Decay phase
    envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)

    # Apply envelope
    waveform *= envelope
    return waveform

# Play music using fractal rhythm and raga notes
def play_music_with_fractal_rhythm(raga, root_frequency=440.0, tempo=120):
    print(f"Playing Raga: {raga['name']} with intervals {raga['intervals']} at tempo {tempo} BPM")
    sample_rate = 44100
    core_pattern = generate_random_core_pattern()
    fractal_rhythm = generate_fractal_rhythm(core_pattern, depth=3)

    current_note = np.random.choice(raga["intervals"])  # Start with a random interval
    next_core_change = datetime.now() + timedelta(minutes=5)  # Schedule core pattern change

    # Calculate beat duration based on tempo
    beat_duration = 60 / tempo

    for step in fractal_rhythm:
        if datetime.now() >= next_core_change:
            # Change the core pattern every 5 minutes
            core_pattern = generate_random_core_pattern()
            fractal_rhythm = generate_fractal_rhythm(core_pattern, depth=3)
            next_core_change = datetime.now() + timedelta(minutes=5)

        if step == 0:
            time.sleep(beat_duration)  # Rest for one beat
            continue

        # Select next note stochastically
        next_note = select_next_note(current_note, raga["intervals"])
        frequency = root_frequency * (2 ** (next_note / 12))

        # Generate waveform for the note with Attack/Decay envelope
        waveform = generate_sine_wave_with_envelope(frequency, beat_duration, sample_rate)

        # Convert to 16-bit PCM
        waveform = (waveform * 32767).astype(np.int16)

        # Play the waveform
        play_obj = sa.play_buffer(waveform, 1, 2, sample_rate)
        play_obj.wait_done()

        current_note = next_note  # Update current note

# Run the dynamic music box
def run_music_box():
    while True:
        raga = choose_raga()
        if raga:
            play_music_with_fractal_rhythm(raga, root_frequency=220.0, tempo=300)  # Example tempo: 90 BPM
        else:
            print("No raga found for the current time.")
            time.sleep(60)  # Retry after 1 minute

# Start the program
if __name__ == "__main__":
    run_music_box()
