import numpy as np
from audio_dsp.utils import wav_io as wavfile

# Audio settings
SAMPLE_RATE = 44100  # Hz
DURATION = 0.2  # Shorter duration for techno snap
BPM = 140  # Faster tempo for techno vibe
BEAT_DURATION = 60 / BPM  # Seconds per beat
GRID_DURATION = 4 * BEAT_DURATION  # 4/4 measure duration (4 beats at 140 BPM)

# Helper functions
def normalize(audio):
    """Normalize audio to 16-bit range with slight distortion for techno grit."""
    audio = np.tanh(audio * np.random.uniform(0.9, 1.1))  # Soft clipping for techno edge
    audio = np.clip(audio, -1, 1)
    return (audio * 32767).astype(np.int16)

def square_wave(freq, duration, sample_rate, pwm=0.5):
    """Generate a square wave with pulse-width modulation."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    phase = np.sin(2 * np.pi * freq * t)
    return np.where(phase > np.sin(np.pi * pwm), 1, -1)  # Variable pulse width

def noise(duration, sample_rate):
    """Generate white noise with random coloring."""
    wave = np.random.uniform(-1, 1, int(sample_rate * duration))
    if np.random.random() > 0.5:  # Randomly apply high-pass filter
        wave = wave - np.roll(wave, 1) * 0.5
    return wave

# Instrument generators with enhanced randomness and techno flavor
def generate_kick():
    """Punchy low-frequency square wave with random detune and sharp decay."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    start_freq = np.random.uniform(70, 130)  # Wider frequency range
    end_freq = np.random.uniform(30, 70)
    freq = np.linspace(start_freq, end_freq, len(t))
    detune = np.random.uniform(-10, 10)  # Slight detune for techno grit
    wave = square_wave(freq + detune, DURATION, SAMPLE_RATE, pwm=np.random.uniform(0.4, 0.6))
    envelope = np.exp(-np.random.uniform(5, 8) * t / DURATION)  # Sharper decay
    return wave * envelope

def generate_snare():
    """Metallic noise with square wave, random timbre and sharp attack."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    noise_weight = np.random.uniform(0.4, 0.8)
    square_freq = np.random.uniform(150, 250)
    noise_part = noise(DURATION, SAMPLE_RATE) * noise_weight
    square_part = square_wave(square_freq, DURATION, SAMPLE_RATE, pwm=np.random.uniform(0.3, 0.7)) * (1 - noise_weight)
    envelope = np.exp(-np.random.uniform(6, 9) * t / DURATION)  # Sharper decay
    return (noise_part + square_part) * envelope

def generate_cymbal():
    """Bright, metallic noise with random filter and fast decay."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    wave = noise(DURATION, SAMPLE_RATE)
    filter_strength = np.random.uniform(0.3, 0.8)  # Random filter for metallic sheen
    wave = wave * filter_strength + np.roll(wave, 1) * (1 - filter_strength)
    envelope = np.exp(-np.random.uniform(8, 12) * t / DURATION)  # Very fast decay
    return wave * envelope

def generate_blip():
    """Sharp, high-pitched square wave with random pitch and PWM."""
    blip_duration = DURATION / np.random.uniform(1.8, 3.0)
    t = np.linspace(0, blip_duration, int(SAMPLE_RATE * blip_duration), endpoint=False)
    pitch = np.random.uniform(600, 1000)  # Wider pitch range
    wave = square_wave(pitch, blip_duration, SAMPLE_RATE, pwm=np.random.uniform(0.2, 0.8))
    envelope = np.exp(-np.random.uniform(10, 15) * t / blip_duration)
    return np.pad(wave * envelope, (0, int(SAMPLE_RATE * (DURATION - blip_duration))))

def generate_swoosh():
    """Fast frequency sweep with random PWM and direction."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    start_freq = np.random.uniform(600, 1400)
    end_freq = np.random.uniform(100, 400)
    if np.random.random() > 0.5:  # Random sweep direction
        start_freq, end_freq = end_freq, start_freq
    freq = np.linspace(start_freq, end_freq, len(t))
    wave = square_wave(freq, DURATION, SAMPLE_RATE, pwm=np.random.uniform(0.3, 0.7))
    envelope = np.exp(-np.random.uniform(3, 5) * t / DURATION)
    return wave * envelope

def generate_fast_arp():
    """Fast arpeggio with random notes and PWM."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    base_notes = [440, 523.25, 659.25, 783.99, 880]  # C4, E4, G4, A4, C5
    notes = np.random.choice(base_notes, size=np.random.randint(3, 6), replace=True)
    segment_len = len(t) // len(notes)
    wave = np.zeros_like(t)
    for i, freq in enumerate(notes):
        start = i * segment_len
        end = (i + 1) * segment_len if i < len(notes) - 1 else len(t)
        segment_duration = (end - start) / SAMPLE_RATE
        segment_wave = square_wave(freq, segment_duration, SAMPLE_RATE, pwm=np.random.uniform(0.3, 0.7))
        segment_wave = segment_wave[:end - start]
        wave[start:end] = segment_wave
    envelope = np.exp(-np.random.uniform(4, 6) * t / DURATION)
    return wave * envelope

def generate_random_arp():
    """Chaotic arpeggio with random note count and frequencies."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    note_count = np.random.randint(5, 8)
    freqs = np.random.uniform(200, 1000, note_count)  # Wider random frequency range
    segment_len = len(t) // note_count
    wave = np.zeros_like(t)
    for i, freq in enumerate(freqs):
        start = i * segment_len
        end = (i + 1) * segment_len if i < len(freqs) - 1 else len(t)
        segment_duration = (end - start) / SAMPLE_RATE
        segment_wave = square_wave(freq, segment_duration, SAMPLE_RATE, pwm=np.random.uniform(0.2, 0.8))
        segment_wave = segment_wave[:end - start]
        wave[start:end] = segment_wave
    envelope = np.exp(-np.random.uniform(4, 6) * t / DURATION)
    return wave * envelope

def generate_pop():
    """Sharp, metallic pop with random pitch and PWM."""
    pop_duration = DURATION / np.random.uniform(4, 6)
    t = np.linspace(0, pop_duration, int(SAMPLE_RATE * pop_duration), endpoint=False)
    pitch = np.random.uniform(400, 800)
    wave = square_wave(pitch, pop_duration, SAMPLE_RATE, pwm=np.random.uniform(0.2, 0.8))
    envelope = np.exp(-np.random.uniform(15, 20) * t / pop_duration)
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
wavfile.write("8bit_techno_samples.wav", SAMPLE_RATE, output)
