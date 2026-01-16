import numpy as np
from audio_dsp.utils import wav_io as wavfile

# Audio settings
SAMPLE_RATE = 44100  # Hz
DURATION = 0.15  # Short duration for hardcore snap
BPM = 160  # Fast tempo for 90s hardcore techno
BEAT_DURATION = 60 / BPM  # Seconds per beat
GRID_DURATION = 4 * BEAT_DURATION  # 4/4 measure duration (4 beats at 160 BPM)

# Helper functions
def normalize(audio):
    """Normalize with 12-bit quantization and random distortion for 90s grit."""
    distortion = np.random.uniform(1.0, 1.6)  # Random distortion
    audio = np.tanh(audio * distortion)  # Soft clipping
    # Simulate 12-bit quantization
    audio = np.round(audio * 2048) / 2048  # 12-bit resolution (4096 levels)
    audio = np.clip(audio * np.random.uniform(0.8, 1.3), -1, 1)  # Hard clipping
    return (audio * 32767).astype(np.int16)

def saw_wave(freq, duration, sample_rate):
    """Generate sawtooth wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return 2 * (freq * t - np.floor(0.5 + freq * t))  # Sawtooth waveform

def triangle_wave(freq, duration, sample_rate):
    """Generate triangle wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return 2 * np.abs(saw_wave(freq, duration, sample_rate)) - 1

def square_wave(freq, duration, sample_rate, pwm=0.5):
    """Generate square wave with pulse-width modulation."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.sign(np.sin(2 * np.pi * freq * t) - np.sin(np.pi * pwm))
    if np.random.random() > 0.3:  # Random bitcrushing for 90s lo-fi
        crush_factor = np.random.randint(2, 6)
        wave = wave[::crush_factor].repeat(crush_factor)[:len(t)]
    return wave

def blend_waveforms(freq, duration, sample_rate, pwm=0.5):
    """Blend saw, triangle, and square waves for wavetable-like sound."""
    weights = np.random.uniform(0, 1, 3)
    weights /= weights.sum()  # Normalize weights
    saw = saw_wave(freq, duration, sample_rate) * weights[0]
    tri = triangle_wave(freq, duration, sample_rate) * weights[1]
    sqr = square_wave(freq, duration, sample_rate, pwm) * weights[2]
    return saw + tri + sqr

def noise(duration, sample_rate):
    """Generate colored noise with random filtering and bursts."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.random.uniform(-1, 1, int(sample_rate * duration))
    if np.random.random() > 0.4:  # Random resonant filter
        filter_type = np.random.choice(['high', 'low'])
        filter_strength = np.random.uniform(0.2, 0.8)
        if filter_type == 'high':
            wave = wave - np.roll(wave, 1) * filter_strength
        else:
            wave = wave * 0.5 + np.roll(wave, 1) * filter_strength
    if np.random.random() > 0.5:  # Random noise bursts for 90s rave
        burst = np.random.uniform(-1, 1, len(wave)) * np.random.uniform(0.2, 0.6)
        wave += burst * np.sin(2 * np.pi * np.random.uniform(20, 60) * t)
    return wave

# Instrument generators with 90s wavetable and hardcore flavor
def generate_kick():
    """Deep, distorted wavetable kick with random detune and sharp decay."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    start_freq = np.random.uniform(50, 160)  # Wide frequency range
    end_freq = np.random.uniform(15, 70)
    freq = np.linspace(start_freq, end_freq, len(t))
    detune = np.random.uniform(-20, 20)  # Wider detune
    wave = blend_waveforms(freq + detune, DURATION, SAMPLE_RATE, pwm=np.random.uniform(0.1, 0.9))
    envelope = np.exp(-np.random.uniform(7, 12) * t / DURATION)  # Sharper decay
    if np.random.random() > 0.5:  # Random noise layer
        wave += noise(DURATION, SAMPLE_RATE) * np.random.uniform(0.1, 0.4)
    return wave * envelope

def generate_snare():
    """Bright, metallic wavetable snare with random timbre."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    noise_weight = np.random.uniform(0.2, 0.9)
    square_freq = np.random.uniform(100, 350)  # Wider pitch range
    noise_part = noise(DURATION, SAMPLE_RATE) * noise_weight
    square_part = blend_waveforms(square_freq, DURATION, SAMPLE_RATE, pwm=np.random.uniform(0.1, 0.9)) * (1 - noise_weight)
    envelope = np.exp(-np.random.uniform(8, 13) * t / DURATION)
    if np.random.random() > 0.4:  # Random chorus effect
        detune = blend_waveforms(square_freq * 1.02, DURATION, SAMPLE_RATE, pwm=np.random.uniform(0.1, 0.9))
        wave = (square_part + detune * 0.3 + noise_part) / 1.3
    else:
        wave = square_part + noise_part
    return wave * envelope

def generate_cymbal():
    """Shimmering wavetable noise with random filter and fast decay."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    wave = noise(DURATION, SAMPLE_RATE)
    filter_strength = np.random.uniform(0.1, 0.9)  # Wide filter range
    wave = wave * filter_strength + np.roll(wave, 1) * (1 - filter_strength)
    if np.random.random() > 0.5:  # Random wavetable layer
        wave += blend_waveforms(np.random.uniform(800, 1200), DURATION, SAMPLE_RATE, pwm=np.random.uniform(0.1, 0.9)) * 0.3
    envelope = np.exp(-np.random.uniform(10, 16) * t / DURATION)
    return wave * envelope

def generate_blip():
    """Piercing wavetable blip with random pitch and chorus."""
    blip_duration = DURATION / np.random.uniform(2, 5)
    t = np.linspace(0, blip_duration, int(SAMPLE_RATE * blip_duration), endpoint=False)
    pitch = np.random.uniform(400, 1400)  # Wide pitch range
    wave = blend_waveforms(pitch, blip_duration, SAMPLE_RATE, pwm=np.random.uniform(0.1, 0.9))
    envelope = np.exp(-np.random.uniform(14, 20) * t / blip_duration)
    if np.random.random() > 0.5:  # Random chorus
        detune = blend_waveforms(pitch * 1.015, blip_duration, SAMPLE_RATE, pwm=np.random.uniform(0.1, 0.9))
        wave = (wave + detune * 0.4) / 1.4
    if np.random.random() > 0.5:  # Random distortion
        wave = np.tanh(wave * np.random.uniform(1.5, 2.5))
    return np.pad(wave * envelope, (0, int(SAMPLE_RATE * (DURATION - blip_duration))))

def generate_swoosh():
    """Sweeping wavetable with random direction and noise bursts."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    start_freq = np.random.uniform(300, 1800)  # Extreme frequency range
    end_freq = np.random.uniform(40, 600)
    if np.random.random() > 0.5:  # Random sweep direction
        start_freq, end_freq = end_freq, start_freq
    freq = np.linspace(start_freq, end_freq, len(t))
    wave = blend_waveforms(freq, DURATION, SAMPLE_RATE, pwm=np.random.uniform(0.1, 0.9))
    envelope = np.exp(-np.random.uniform(5, 8) * t / DURATION)
    if np.random.random() > 0.4:  # Random noise burst
        wave += noise(DURATION, SAMPLE_RATE) * np.random.uniform(0.1, 0.5)
    return wave * envelope

def generate_fast_arp():
    """Wavetable arpeggio with random notes and chorus."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    base_notes = [440, 523.25, 659.25, 783.99, 880, 1046.50, 1318.51]  # Extended note pool
    notes = np.random.choice(base_notes, size=np.random.randint(3, 8), replace=True)
    segment_len = len(t) // len(notes)
    wave = np.zeros_like(t)
    for i, freq in enumerate(notes):
        start = i * segment_len
        end = (i + 1) * segment_len if i < len(notes) - 1 else len(t)
        segment_samples = end - start
        segment_duration = segment_samples / SAMPLE_RATE
        segment_wave = blend_waveforms(freq, segment_duration, SAMPLE_RATE, pwm=np.random.uniform(0.1, 0.9))
        segment_wave = segment_wave[:segment_samples]
        wave[start:end] = segment_wave
    envelope = np.exp(-np.random.uniform(6, 9) * t / DURATION)
    if np.random.random() > 0.5:  # Random chorus
        detune = np.zeros_like(t)
        for i, freq in enumerate(notes):
            start = i * segment_len
            end = (i + 1) * segment_len if i < len(notes) - 1 else len(t)
            segment_samples = end - start
            segment_duration = segment_samples / SAMPLE_RATE
            segment_detune = blend_waveforms(freq * 1.015, segment_duration, SAMPLE_RATE, pwm=np.random.uniform(0.1, 0.9))
            detune[start:end] = segment_detune[:segment_samples]
        wave = (wave + detune * 0.3) / 1.3
    return wave * envelope

def generate_random_arp():
    """Chaotic wavetable arpeggio with extreme frequency jumps."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    note_count = np.random.randint(5, 10)
    freqs = np.random.uniform(100, 1400, note_count)  # Extreme frequency range
    segment_len = len(t) // note_count
    wave = np.zeros_like(t)
    for i, freq in enumerate(freqs):
        start = i * segment_len
        end = (i + 1) * segment_len if i < len(freqs) - 1 else len(t)
        segment_samples = end - start
        segment_duration = segment_samples / SAMPLE_RATE
        segment_wave = blend_waveforms(freq, segment_duration, SAMPLE_RATE, pwm=np.random.uniform(0.1, 0.9))
        segment_wave = segment_wave[:segment_samples]
        wave[start:end] = segment_wave
    envelope = np.exp(-np.random.uniform(6, 9) * t / DURATION)
    return wave * envelope

def generate_pop():
    """Sharp wavetable pop with random pitch and distortion."""
    pop_duration = DURATION / np.random.uniform(5, 8)
    t = np.linspace(0, pop_duration, int(SAMPLE_RATE * pop_duration), endpoint=False)
    pitch = np.random.uniform(200, 1200)
    wave = blend_waveforms(pitch, pop_duration, SAMPLE_RATE, pwm=np.random.uniform(0.1, 0.9))
    envelope = np.exp(-np.random.uniform(20, 30) * t / pop_duration)
    if np.random.random() > 0.5:  # Random distortion
        wave = np.tanh(wave * np.random.uniform(1.5, 3.0))
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
wavfile.write("90s_wavetable_hc_samples.wav", SAMPLE_RATE, output)
