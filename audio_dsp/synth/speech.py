import numpy as np
from audio_dsp.utils import wav_io as wavfile

# Audio settings
SAMPLE_RATE = 44100  # Hz
DURATION = 0.2  # Duration per phoneme for clarity
BPM = 120  # Default tempo (user can specify)
BEAT_DURATION = 60 / BPM  # Seconds per beat
GRID_DURATION = 4 * BEAT_DURATION  # 4/4 measure duration

# Helper functions
def normalize(audio):
    """Normalize audio to 16-bit range with light compression."""
    audio = np.tanh(audio * np.random.uniform(0.8, 1.2))  # Light compression
    audio = np.clip(audio, -1, 1)
    return (audio * 32767).astype(np.int16)

def sine_wave(freq, duration, sample_rate):
    """Generate sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)

def noise(duration, sample_rate, filter_type='band'):
    """Generate filtered noise for consonants."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.random.uniform(-1, 1, int(sample_rate * duration))
    if filter_type == 'high':
        wave = wave - np.roll(wave, 1) * np.random.uniform(0.5, 0.8)
    elif filter_type == 'low':
        wave = wave * 0.5 + np.roll(wave, 1) * np.random.uniform(0.3, 0.7)
    else:  # Bandpass
        filter_freq = np.random.uniform(1000, 4000)
        filter_mod = np.sin(2 * np.pi * filter_freq * t) * 0.5 + 0.5
        wave *= filter_mod
    return wave

# Phoneme generators
def generate_a():
    """Vowel 'A': Formant synthesis with two formants."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = np.random.uniform(100, 150)
    formant1 = sine_wave(np.random.uniform(600, 800), DURATION, SAMPLE_RATE) * 0.6
    formant2 = sine_wave(np.random.uniform(1000, 1200), DURATION, SAMPLE_RATE) * 0.3
    wave = sine_wave(freq, DURATION, SAMPLE_RATE) * 0.4 + formant1 + formant2
    envelope = np.exp(-np.random.uniform(3, 5) * t / DURATION)
    return wave * envelope

def generate_b():
    """Consonant 'B': Plosive with noise burst and low tone."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = np.random.uniform(80, 120)
    wave = sine_wave(freq, DURATION, SAMPLE_RATE) * 0.3
    noise_part = noise(DURATION / 2, SAMPLE_RATE, 'band') * np.random.uniform(0.4, 0.6)
    noise_part = np.pad(noise_part, (0, len(t) - len(noise_part)))
    envelope = np.exp(-np.random.uniform(6, 9) * t / DURATION)
    return (wave + noise_part) * envelope

def generate_c():
    """Consonant 'C': Hard 'K' or soft 'S' sound, randomized."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    if np.random.random() > 0.5:  # Hard 'K'-like
        freq = np.random.uniform(100, 150)
        wave = sine_wave(freq, DURATION, SAMPLE_RATE) * 0.3
        noise_part = noise(DURATION / 2, SAMPLE_RATE, 'band') * np.random.uniform(0.5, 0.7)
        noise_part = np.pad(noise_part, (0, len(t) - len(noise_part)))
        envelope = np.exp(-np.random.uniform(6, 9) * t / DURATION)
    else:  # Soft 'S'-like
        wave = noise(DURATION, SAMPLE_RATE, 'high') * 0.8
        envelope = np.exp(-np.random.uniform(5, 8) * t / DURATION)
    return wave * envelope

def generate_d():
    """Consonant 'D': Sharp plosive with noise and mid tone."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = np.random.uniform(100, 150)
    wave = sine_wave(freq, DURATION, SAMPLE_RATE) * 0.3
    noise_part = noise(DURATION / 2, SAMPLE_RATE, 'band') * np.random.uniform(0.3, 0.5)
    noise_part = np.pad(noise_part, (0, len(t) - len(noise_part)))
    envelope = np.exp(-np.random.uniform(7, 10) * t / DURATION)
    return (wave + noise_part) * envelope

def generate_e():
    """Vowel 'E': Higher formants for brighter sound."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = np.random.uniform(120, 180)
    formant1 = sine_wave(np.random.uniform(400, 600), DURATION, SAMPLE_RATE) * 0.5
    formant2 = sine_wave(np.random.uniform(2000, 2400), DURATION, SAMPLE_RATE) * 0.3
    wave = sine_wave(freq, DURATION, SAMPLE_RATE) * 0.4 + formant1 + formant2
    envelope = np.exp(-np.random.uniform(3, 5) * t / DURATION)
    return wave * envelope

def generate_f():
    """Consonant 'F': Hiss with high-frequency noise."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    wave = noise(DURATION, SAMPLE_RATE, 'high') * 0.8
    envelope = np.exp(-np.random.uniform(5, 8) * t / DURATION)
    return wave * envelope

def generate_g():
    """Consonant 'G': Plosive with low-mid tone and noise."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = np.random.uniform(90, 140)
    wave = sine_wave(freq, DURATION, SAMPLE_RATE) * 0.3
    noise_part = noise(DURATION / 2, SAMPLE_RATE, 'band') * np.random.uniform(0.4, 0.6)
    noise_part = np.pad(noise_part, (0, len(t) - len(noise_part)))
    envelope = np.exp(-np.random.uniform(6, 9) * t / DURATION)
    return (wave + noise_part) * envelope

def generate_h():
    """Consonant 'H': Breathy noise with soft tone."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    wave = sine_wave(np.random.uniform(100, 150), DURATION, SAMPLE_RATE) * 0.2
    noise_part = noise(DURATION, SAMPLE_RATE, 'high') * np.random.uniform(0.5, 0.7)
    envelope = np.exp(-np.random.uniform(5, 8) * t / DURATION)
    return (wave + noise_part) * envelope

def generate_i():
    """Vowel 'I': High formants for sharp sound."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = np.random.uniform(130, 190)
    formant1 = sine_wave(np.random.uniform(250, 350), DURATION, SAMPLE_RATE) * 0.5
    formant2 = sine_wave(np.random.uniform(2200, 2600), DURATION, SAMPLE_RATE) * 0.3
    wave = sine_wave(freq, DURATION, SAMPLE_RATE) * 0.4 + formant1 + formant2
    envelope = np.exp(-np.random.uniform(3, 5) * t / DURATION)
    return wave * envelope

def generate_j():
    """Consonant 'J': Soft plosive with noise and mid tone."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = np.random.uniform(110, 160)
    wave = sine_wave(freq, DURATION, SAMPLE_RATE) * 0.3
    noise_part = noise(DURATION / 2, SAMPLE_RATE, 'band') * np.random.uniform(0.3, 0.5)
    noise_part = np.pad(noise_part, (0, len(t) - len(noise_part)))
    envelope = np.exp(-np.random.uniform(6, 9) * t / DURATION)
    return (wave + noise_part) * envelope

def generate_k():
    """Consonant 'K': Hard plosive with noise and low-mid tone."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = np.random.uniform(100, 150)
    wave = sine_wave(freq, DURATION, SAMPLE_RATE) * 0.3
    noise_part = noise(DURATION / 2, SAMPLE_RATE, 'band') * np.random.uniform(0.5, 0.7)
    noise_part = np.pad(noise_part, (0, len(t) - len(noise_part)))
    envelope = np.exp(-np.random.uniform(6, 9) * t / DURATION)
    return (wave + noise_part) * envelope

def generate_l():
    """Consonant 'L': Liquid sound with low formant."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = np.random.uniform(100, 150)
    formant1 = sine_wave(np.random.uniform(400, 600), DURATION, SAMPLE_RATE) * 0.4
    wave = sine_wave(freq, DURATION, SAMPLE_RATE) * 0.4 + formant1
    envelope = np.exp(-np.random.uniform(4, 6) * t / DURATION)
    return wave * envelope

def generate_m():
    """Consonant 'M': Nasal sound with low formants."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = np.random.uniform(80, 120)
    formant1 = sine_wave(np.random.uniform(300, 500), DURATION, SAMPLE_RATE) * 0.5
    formant2 = sine_wave(np.random.uniform(1000, 1300), DURATION, SAMPLE_RATE) * 0.3
    wave = sine_wave(freq, DURATION, SAMPLE_RATE) * 0.4 + formant1 + formant2
    envelope = np.exp(-np.random.uniform(4, 6) * t / DURATION)
    return wave * envelope

def generate_n():
    """Consonant 'N': Nasal sound with mid formants."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = np.random.uniform(90, 140)
    formant1 = sine_wave(np.random.uniform(400, 600), DURATION, SAMPLE_RATE) * 0.5
    formant2 = sine_wave(np.random.uniform(1200, 1500), DURATION, SAMPLE_RATE) * 0.3
    wave = sine_wave(freq, DURATION, SAMPLE_RATE) * 0.4 + formant1 + formant2
    envelope = np.exp(-np.random.uniform(4, 6) * t / DURATION)
    return wave * envelope

def generate_o():
    """Vowel 'O': Lower formants for rounded sound."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = np.random.uniform(90, 140)
    formant1 = sine_wave(np.random.uniform(400, 600), DURATION, SAMPLE_RATE) * 0.6
    formant2 = sine_wave(np.random.uniform(800, 1000), DURATION, SAMPLE_RATE) * 0.3
    wave = sine_wave(freq, DURATION, SAMPLE_RATE) * 0.4 + formant1 + formant2
    envelope = np.exp(-np.random.uniform(3, 5) * t / DURATION)
    return wave * envelope

def generate_p():
    """Consonant 'P': Soft plosive with noise and low tone."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = np.random.uniform(80, 120)
    wave = sine_wave(freq, DURATION, SAMPLE_RATE) * 0.3
    noise_part = noise(DURATION / 2, SAMPLE_RATE, 'band') * np.random.uniform(0.4, 0.6)
    noise_part = np.pad(noise_part, (0, len(t) - len(noise_part)))
    envelope = np.exp(-np.random.uniform(6, 9) * t / DURATION)
    return (wave + noise_part) * envelope

def generate_q():
    """Consonant 'Q': 'K'-like plosive with unique formant."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = np.random.uniform(100, 150)
    wave = sine_wave(freq, DURATION, SAMPLE_RATE) * 0.3
    noise_part = noise(DURATION / 2, SAMPLE_RATE, 'band') * np.random.uniform(0.5, 0.7)
    noise_part = np.pad(noise_part, (0, len(t) - len(noise_part)))
    formant = sine_wave(np.random.uniform(800, 1000), DURATION, SAMPLE_RATE) * 0.2
    envelope = np.exp(-np.random.uniform(6, 9) * t / DURATION)
    return (wave + noise_part + formant) * envelope

def generate_r():
    """Consonant 'R': Rolled sound with modulated tone."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = np.random.uniform(100, 150)
    wave = sine_wave(freq, DURATION, SAMPLE_RATE) * 0.4
    modulation = np.sin(2 * np.pi * np.random.uniform(5, 10) * t) * 0.3  # Vibrato for roll
    wave += modulation * sine_wave(np.random.uniform(300, 500), DURATION, SAMPLE_RATE)
    envelope = np.exp(-np.random.uniform(4, 6) * t / DURATION)
    return wave * envelope

def generate_s():
    """Consonant 'S': Hiss with high-frequency noise."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    wave = noise(DURATION, SAMPLE_RATE, 'high')
    envelope = np.exp(-np.random.uniform(5, 8) * t / DURATION)
    return wave * envelope

def generate_t():
    """Consonant 'T': Sharp click with noise and high tone."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = np.random.uniform(120, 180)
    wave = sine_wave(freq, DURATION, SAMPLE_RATE) * 0.3
    noise_part = noise(DURATION / 2, SAMPLE_RATE, 'band') * np.random.uniform(0.4, 0.6)
    noise_part = np.pad(noise_part, (0, len(t) - len(noise_part)))
    envelope = np.exp(-np.random.uniform(7, 10) * t / DURATION)
    return (wave + noise_part) * envelope

def generate_u():
    """Vowel 'U': Deep formants for dark sound."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = np.random.uniform(80, 130)
    formant1 = sine_wave(np.random.uniform(300, 400), DURATION, SAMPLE_RATE) * 0.6
    formant2 = sine_wave(np.random.uniform(600, 800), DURATION, SAMPLE_RATE) * 0.3
    wave = sine_wave(freq, DURATION, SAMPLE_RATE) * 0.4 + formant1 + formant2
    envelope = np.exp(-np.random.uniform(3, 5) * t / DURATION)
    return wave * envelope

def generate_v():
    """Consonant 'V': Voiced fricative with noise and tone."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = np.random.uniform(90, 140)
    wave = sine_wave(freq, DURATION, SAMPLE_RATE) * 0.4
    noise_part = noise(DURATION, SAMPLE_RATE, 'high') * np.random.uniform(0.3, 0.5)
    envelope = np.exp(-np.random.uniform(5, 8) * t / DURATION)
    return (wave + noise_part) * envelope

def generate_w():
    """Consonant 'W': Glide with low formants."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = np.random.uniform(80, 120)
    formant1 = sine_wave(np.random.uniform(300, 500), DURATION, SAMPLE_RATE) * 0.5
    formant2 = sine_wave(np.random.uniform(700, 900), DURATION, SAMPLE_RATE) * 0.3
    wave = sine_wave(freq, DURATION, SAMPLE_RATE) * 0.4 + formant1 + formant2
    envelope = np.exp(-np.random.uniform(4, 6) * t / DURATION)
    return wave * envelope

def generate_x():
    """Consonant 'X': Harsh fricative like 'ks'."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    wave = noise(DURATION, SAMPLE_RATE, 'high') * 0.8
    noise_part = noise(DURATION / 2, SAMPLE_RATE, 'band') * np.random.uniform(0.3, 0.5)
    noise_part = np.pad(noise_part, (0, len(t) - len(noise_part)))
    envelope = np.exp(-np.random.uniform(5, 8) * t / DURATION)
    return (wave + noise_part) * envelope

def generate_y():
    """Consonant 'Y': Glide with high formants."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = np.random.uniform(110, 160)
    formant1 = sine_wave(np.random.uniform(400, 600), DURATION, SAMPLE_RATE) * 0.5
    formant2 = sine_wave(np.random.uniform(2000, 2400), DURATION, SAMPLE_RATE) * 0.3
    wave = sine_wave(freq, DURATION, SAMPLE_RATE) * 0.4 + formant1 + formant2
    envelope = np.exp(-np.random.uniform(4, 6) * t / DURATION)
    return wave * envelope

def generate_z():
    """Consonant 'Z': Voiced fricative with noise and tone."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = np.random.uniform(90, 140)
    wave = sine_wave(freq, DURATION, SAMPLE_RATE) * 0.4
    noise_part = noise(DURATION, SAMPLE_RATE, 'high') * np.random.uniform(0.3, 0.5)
    envelope = np.exp(-np.random.uniform(5, 8) * t / DURATION)
    return (wave + noise_part) * envelope

def generate_default():
    """Default sound: Short neutral tone."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    freq = np.random.uniform(150, 250)
    wave = sine_wave(freq, DURATION, SAMPLE_RATE)
    envelope = np.exp(-np.random.uniform(5, 8) * t / DURATION)
    return wave * envelope

# Character-to-sound mapping
CHAR_MAP = {
    'a': generate_a, 'b': generate_b, 'c': generate_c, 'd': generate_d, 'e': generate_e,
    'f': generate_f, 'g': generate_g, 'h': generate_h, 'i': generate_i, 'j': generate_j,
    'k': generate_k, 'l': generate_l, 'm': generate_m, 'n': generate_n, 'o': generate_o,
    'p': generate_p, 'q': generate_q, 'r': generate_r, 's': generate_s, 't': generate_t,
    'u': generate_u, 'v': generate_v, 'w': generate_w, 'x': generate_x, 'y': generate_y,
    'z': generate_z
}

# Main function
def generate_speech_synth(text, bpm=120):
    global BPM, BEAT_DURATION, GRID_DURATION
    BPM = bpm
    BEAT_DURATION = 60 / BPM
    GRID_DURATION = 4 * BEAT_DURATION
    text = text.lower()[:8]  # Limit to 8 characters
    samples = []
    for char in text:
        generator = CHAR_MAP.get(char, generate_default)
        samples.append(generator())
    while len(samples) < 8:
        samples.append(generate_default())
    # Create 4/4 beat grid
    output = np.zeros(int(SAMPLE_RATE * GRID_DURATION))
    beat_samples = int(SAMPLE_RATE * BEAT_DURATION / 2)  # 8th note spacing
    for i, sample in enumerate(samples):
        start = i * beat_samples
        if start + len(sample) <= len(output):
            output[start:start + len(sample)] += sample
    # Normalize and save
    output = normalize(output)
    wavfile.write("speech_synth.wav", SAMPLE_RATE, output)

# Example usage
if __name__ == "__main__":
    input_text = input("Enter a string (up to 8 characters): ")
    bpm = int(input("Enter BPM (e.g., 120): "))
    generate_speech_synth(input_text, bpm)
