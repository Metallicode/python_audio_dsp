import numpy as np
from audio_dsp.utils import wav_io as wavfile

# Audio settings
SAMPLE_RATE = 44100  # Hz
DURATION = 0.15  # Very short duration for hardcore snap
BPM = 160  # Faster tempo for hardcore techno
BEAT_DURATION = 60 / BPM  # Seconds per beat
GRID_DURATION = 4 * BEAT_DURATION  # 4/4 measure duration (4 beats at 160 BPM)

# Helper functions
def normalize(audio):
    """Normalize with hard clipping and random distortion for hardcore grit."""
    distortion = np.random.uniform(1.0, 1.5)  # Random distortion intensity
    audio = np.tanh(audio * distortion)  # Soft clipping
    audio = np.clip(audio * np.random.uniform(0.8, 1.2), -1, 1)  # Hard clipping with random gain
    return (audio * 32767).astype(np.int16)

def square_wave(freq, duration, sample_rate, pwm=0.5):
    """Generate square wave with pulse-width modulation and random bitcrushing."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.sign(np.sin(2 * np.pi * freq * t) - np.sin(np.pi * pwm))
    if np.random.random() > 0.3:  # Random bitcrushing
        crush_factor = np.random.randint(2, 8)
        wave = wave[::crush_factor].repeat(crush_factor)[:len(t)]
    return wave

def noise(duration, sample_rate):
    """Generate colored noise with random filtering and bursts."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.random.uniform(-1, 1, int(sample_rate * duration))
    if np.random.random() > 0.4:  # Random high-pass or low-pass
        filter_type = np.random.choice(['high', 'low'])
        if filter_type == 'high':
            wave = wave - np.roll(wave, 1) * np.random.uniform(0.3, 0.7)
        else:
            wave = wave * 0.5 + np.roll(wave, 1) * np.random.uniform(0.3, 0.7)
    if np.random.random() > 0.6:  # Random noise bursts
        burst = np.random.uniform(-1, 1, len(wave)) * np.random.uniform(0.2, 0.5)
        wave += burst * np.sin(2 * np.pi * np.random.uniform(10, 50) * t)
    return wave

# Instrument generators with extreme randomness and hardcore flavor
def generate_kick():
    """Heavy, distorted square wave with random detune and aggressive decay."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    start_freq = np.random.uniform(60, 150)  # Wider frequency range
    end_freq = np.random.uniform(20, 80)
    freq = np.linspace(start_freq, end_freq, len(t))
    detune = np.random.uniform(-15, 15)  # More detune
    wave = square_wave(freq + detune, DURATION, SAMPLE_RATE, pwm=np.random.uniform(0.2, 0.8))
    envelope = np.exp(-np.random.uniform(6, 10) * t / DURATION)  # Very sharp decay
    if np.random.random() > 0.5:  # Random noise layer
        wave += noise(DURATION, SAMPLE_RATE) * np.random.uniform(0.1, 0.3)
    return wave * envelope

def generate_snare():
    """Harsh noise with square wave, random metallic timbre and distortion."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    noise_weight = np.random.uniform(0.3, 0.9)
    square_freq = np.random.uniform(120, 300)  # Wider pitch range
    noise_part = noise(DURATION, SAMPLE_RATE) * noise_weight
    square_part = square_wave(square_freq, DURATION, SAMPLE_RATE, pwm=np.random.uniform(0.1, 0.9)) * (1 - noise_weight)
    envelope = np.exp(-np.random.uniform(7, 11) * t / DURATION)  # Sharper decay
    return (noise_part + square_part) * envelope

def generate_cymbal():
    """Piercing noise with random filter and extreme decay."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    wave = noise(DURATION, SAMPLE_RATE)
    filter_strength = np.random.uniform(0.2, 0.9)  # Wider filter range
    wave = wave * filter_strength + np.roll(wave, 1) * (1 - filter_strength)
    envelope = np.exp(-np.random.uniform(9, 14) * t / DURATION)  # Extremely fast decay
    return wave * envelope

def generate_blip():
    """Chaotic high-pitched square with random PWM and duration."""
    blip_duration = DURATION / np.random.uniform(2, 4)
    t = np.linspace(0, blip_duration, int(SAMPLE_RATE * blip_duration), endpoint=False)
    pitch = np.random.uniform(500, 1200)  # Wider pitch range
    wave = square_wave(pitch, blip_duration, SAMPLE_RATE, pwm=np.random.uniform(0.1, 0.9))
    envelope = np.exp(-np.random.uniform(12, 18) * t / blip_duration)
    if np.random.random() > 0.5:  # Random distortion
        wave = np.tanh(wave * np.random.uniform(1.2, 2.0))
    return np.pad(wave * envelope, (0, int(SAMPLE_RATE * (DURATION - blip_duration))))

def generate_swoosh():
    """Wild frequency sweep with random direction and harsh PWM."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    start_freq = np.random.uniform(400, 1600)  # Extreme frequency range
    end_freq = np.random.uniform(50, 500)
    if np.random.random() > 0.5:  # Random sweep direction
        start_freq, end_freq = end_freq, start_freq
    freq = np.linspace(start_freq, end_freq, len(t))
    wave = square_wave(freq, DURATION, SAMPLE_RATE, pwm=np.random.uniform(0.1, 0.9))
    envelope = np.exp(-np.random.uniform(4, 7) * t / DURATION)
    if np.random.random() > 0.4:  # Random noise burst
        wave += noise(DURATION, SAMPLE_RATE) * np.random.uniform(0.1, 0.4)
    return wave * envelope

def generate_fast_arp():
    """Chaotic arpeggio with random notes, PWM, and distortion."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    base_notes = [440, 523.25, 659.25, 783.99, 880, 1046.50]  # Extended note pool
    notes = np.random.choice(base_notes, size=np.random.randint(3, 7), replace=True)
    segment_len = len(t) // len(notes)
    wave = np.zeros_like(t)
    for i, freq in enumerate(notes):
        start = i * segment_len
        end = (i + 1) * segment_len if i < len(notes) - 1 else len(t)
        segment_samples = end - start
        segment_duration = segment_samples / SAMPLE_RATE
        segment_wave = square_wave(freq, segment_duration, SAMPLE_RATE, pwm=np.random.uniform(0.1, 0.9))
        segment_wave = segment_wave[:segment_samples]  # Ensure exact length
        wave[start:end] = segment_wave
    envelope = np.exp(-np.random.uniform(5, 8) * t / DURATION)
    if np.random.random() > 0.5:  # Random distortion
        wave = np.tanh(wave * np.random.uniform(1.2, 2.0))
    return wave * envelope

def generate_random_arp():
    """Insane arpeggio with extreme frequency range and random jumps."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    note_count = np.random.randint(5, 9)
    freqs = np.random.uniform(150, 1200, note_count)  # Extreme frequency range
    segment_len = len(t) // note_count
    wave = np.zeros_like(t)
    for i, freq in enumerate(freqs):
        start = i * segment_len
        end = (i + 1) * segment_len if i < len(freqs) - 1 else len(t)
        segment_samples = end - start
        segment_duration = segment_samples / SAMPLE_RATE
        segment_wave = square_wave(freq, segment_duration, SAMPLE_RATE, pwm=np.random.uniform(0.1, 0.9))
        segment_wave = segment_wave[:segment_samples]  # Ensure exact length
        wave[start:end] = segment_wave
    envelope = np.exp(-np.random.uniform(5, 8) * t / DURATION)
    return wave * envelope

def generate_pop():
    """Ultra-sharp pop with random pitch and extreme PWM."""
    pop_duration = DURATION / np.random.uniform(5, 8)
    t = np.linspace(0, pop_duration, int(SAMPLE_RATE * pop_duration), endpoint=False)
    pitch = np.random.uniform(300, 1000)
    wave = square_wave(pitch, pop_duration, SAMPLE_RATE, pwm=np.random.uniform(0.1, 0.9))
    envelope = np.exp(-np.random.uniform(18, 25) * t / pop_duration)
    if np.random.random() > 0.5:  # Random distortion
        wave = np.tanh(wave * np.random.uniform(1.5, 2.5))
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
wavfile.write("8bit_hc_samples.wav", SAMPLE_RATE, output)
