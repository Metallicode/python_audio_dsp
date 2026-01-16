import numpy as np
from audio_dsp.utils import wav_io as wavfile

# Audio settings
SAMPLE_RATE = 44100  # Hz
DURATION = 0.25  # Duration of each sample in seconds
BPM = 120  # Beats per minute
BEAT_DURATION = 60 / BPM  # Seconds per beat
GRID_DURATION = 2.0  # 4/4 measure duration (4 beats at 120 BPM)

# Helper functions
def normalize(audio):
    """Normalize audio to 16-bit range."""
    audio = np.clip(audio, -1, 1)
    return (audio * 32767).astype(np.int16)

def square_wave(freq, duration, sample_rate):
    """Generate a square wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sign(np.sin(2 * np.pi * freq * t))

def noise(duration, sample_rate):
    """Generate white noise."""
    return np.random.uniform(-1, 1, int(sample_rate * duration))

# Instrument generators with randomness
def generate_kick():
    """Low-frequency decaying square wave with random frequency range."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    start_freq = np.random.uniform(80, 120)  # Random start frequency
    end_freq = np.random.uniform(40, 60)     # Random end frequency
    freq = np.linspace(start_freq, end_freq, len(t))
    wave = square_wave(freq, DURATION, SAMPLE_RATE)
    envelope = np.exp(-np.random.uniform(4, 6) * t / DURATION)  # Random decay
    return wave * envelope

def generate_snare():
    """Noise with a short square wave, random balance and decay."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    noise_weight = np.random.uniform(0.5, 0.9)  # Random noise/square balance
    square_freq = np.random.uniform(180, 220)   # Random square wave pitch
    noise_part = noise(DURATION, SAMPLE_RATE) * noise_weight
    square_part = square_wave(square_freq, DURATION, SAMPLE_RATE) * (1 - noise_weight)
    envelope = np.exp(-np.random.uniform(5, 7) * t / DURATION)  # Random decay
    return (noise_part + square_part) * envelope

def generate_cymbal():
    """High-frequency noise with random decay and filter."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    wave = noise(DURATION, SAMPLE_RATE)
    envelope = np.exp(-np.random.uniform(7, 9) * t / DURATION)  # Random decay
    # Simple low-pass filter effect with random cutoff
    filter_strength = np.random.uniform(0.5, 1.0)
    wave = wave * filter_strength + np.roll(wave, 1) * (1 - filter_strength)
    return wave * envelope

def generate_blip():
    """Short high-pitched square wave with random pitch and duration."""
    blip_duration = DURATION / np.random.uniform(1.5, 2.5)  # Random short duration
    t = np.linspace(0, blip_duration, int(SAMPLE_RATE * blip_duration), endpoint=False)
    pitch = np.random.uniform(700, 900)  # Random high pitch
    wave = square_wave(pitch, blip_duration, SAMPLE_RATE)
    envelope = np.exp(-np.random.uniform(8, 12) * t / blip_duration)  # Random decay
    return np.pad(wave * envelope, (0, int(SAMPLE_RATE * (DURATION - blip_duration))))

def generate_swoosh():
    """Sweeping square wave with random frequency range."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    start_freq = np.random.uniform(800, 1200)  # Random start frequency
    end_freq = np.random.uniform(150, 250)     # Random end frequency
    freq = np.linspace(start_freq, end_freq, len(t))
    wave = square_wave(freq, DURATION, SAMPLE_RATE)
    envelope = np.exp(-np.random.uniform(2.5, 3.5) * t / DURATION)  # Random decay
    return wave * envelope

def generate_fast_arp():
    """Fast up/down arpeggio with random note selection."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    base_notes = [440, 554.37, 659.25, 880]  # C4, E4, G4, C5
    notes = np.random.choice(base_notes, size=4, replace=True)  # Random note order
    segment_len = len(t) // len(notes)
    wave = np.zeros_like(t)
    for i, freq in enumerate(notes):
        start = i * segment_len
        end = (i + 1) * segment_len if i < len(notes) - 1 else len(t)
        segment_duration = (end - start) / SAMPLE_RATE  # Exact duration for segment
        segment_wave = square_wave(freq, segment_duration, SAMPLE_RATE)
        segment_wave = segment_wave[:end - start]  # Ensure exact length
        wave[start:end] = segment_wave
    envelope = np.exp(-np.random.uniform(3.5, 4.5) * t / DURATION)  # Random decay
    return wave * envelope

def generate_random_arp():
    """Random frequency jumps with random note count."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    note_count = np.random.randint(4, 7)  # Random number of notes
    freqs = np.random.choice([220, 330, 440, 550, 660, 880], size=note_count)  # Random notes
    segment_len = len(t) // note_count
    wave = np.zeros_like(t)
    for i, freq in enumerate(freqs):
        start = i * segment_len
        end = (i + 1) * segment_len if i < len(freqs) - 1 else len(t)
        segment_duration = (end - start) / SAMPLE_RATE  # Exact duration for segment
        segment_wave = square_wave(freq, segment_duration, SAMPLE_RATE)
        segment_wave = segment_wave[:end - start]  # Ensure exact length
        wave[start:end] = segment_wave
    envelope = np.exp(-np.random.uniform(3.5, 4.5) * t / DURATION)  # Random decay
    return wave * envelope

def generate_pop():
    """Short, sharp square wave with random pitch and duration."""
    pop_duration = DURATION / np.random.uniform(3, 5)  # Random very short duration
    t = np.linspace(0, pop_duration, int(SAMPLE_RATE * pop_duration), endpoint=False)
    pitch = np.random.uniform(500, 700)  # Random mid-high pitch
    wave = square_wave(pitch, pop_duration, SAMPLE_RATE)
    envelope = np.exp(-np.random.uniform(12, 18) * t / pop_duration)  # Random sharp decay
    return np.pad(wave * envelope, (0, int(SAMPLE_RATE * (DURATION - pop_duration))))

# Generate all samples
samples = [
    generate_kick(),
    generate_snare(),
    generate_cymbal(),
    generate_blip(),
    generate_swoosh(),
    generate_fast_arp(),
    generate_random_arp(),
    generate_pop()
]

# Create 4/4 beat grid (8 samples over 4 beats)
output = np.zeros(int(SAMPLE_RATE * GRID_DURATION))
beat_samples = int(SAMPLE_RATE * BEAT_DURATION / 2)  # 8th note spacing
for i, sample in enumerate(samples):
    start = i * beat_samples
    if start + len(sample) <= len(output):
        output[start:start + len(sample)] += sample

# Normalize and save
output = normalize(output)
wavfile.write("8bit_samples.wav", SAMPLE_RATE, output)
