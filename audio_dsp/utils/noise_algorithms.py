import numpy as np
import wave

# Noise Algorithms
def white_noise(length, sample_rate=44100):
    """Generate white noise."""
    num_samples = int(length * sample_rate)
    return np.random.uniform(-1, 1, num_samples)

def pink_noise(length, sample_rate=44100):
    """Generate pink noise using Voss-McCartney algorithm."""
    num_samples = int(length * sample_rate)
    pink = np.random.randn(num_samples)
    b = [0.02109238, 0.07113478, 0.68873558]
    a = [1.0, -1.9067023, 0.91026855]
    pink = np.convolve(pink, b, mode='same') / sum(a)
    return pink / max(np.abs(pink))

def brown_noise(length, sample_rate=44100):
    """Generate Brownian noise."""
    num_samples = int(length * sample_rate)
    brown = np.cumsum(np.random.uniform(-1, 1, num_samples))
    return brown / max(np.abs(brown))

def blue_noise(length, sample_rate=44100):
    """Generate blue noise."""
    num_samples = int(length * sample_rate)
    white = np.random.uniform(-1, 1, num_samples)
    fft = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(num_samples, 1 / sample_rate)
    fft *= np.sqrt(frequencies)
    return np.fft.irfft(fft).real

def violet_noise(length, sample_rate=44100):
    """Generate violet noise."""
    num_samples = int(length * sample_rate)
    white = np.random.uniform(-1, 1, num_samples)
    fft = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(num_samples, 1 / sample_rate)
    fft *= frequencies
    return np.fft.irfft(fft).real

def gaussian_noise(length, sample_rate=44100):
    """Generate Gaussian noise."""
    num_samples = int(length * sample_rate)
    return np.random.normal(0, 0.5, num_samples)

def uniform_random_noise(length, sample_rate=44100):
    """Generate uniform random noise."""
    num_samples = int(length * sample_rate)
    return np.random.uniform(-1, 1, num_samples)

def perlin_noise(length, sample_rate=44100, octaves=1, persistence=0.5, lacunarity=2.0, frequency=1000.0):
    """
    Generate Perlin noise with an adjustable frequency.
    Parameters:
    - frequency: Controls the perceived frequency of the noise (higher = faster changes).
    Requires: pip install noise
    """
    import noise  # Lazy import for optional dependency
    num_samples = int(length * sample_rate)
    _noise = np.zeros(num_samples)

    # Scale input to pnoise1 by the frequency multiplier
    for i in range(num_samples):
        _noise[i] = noise.pnoise1(i * frequency / sample_rate, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
    
    # Normalize to [-1, 1]
    _min, _max = np.min(_noise), np.max(_noise)
    if _max - _min > 0:
        _noise = 2 * (_noise - _min) / (_max - _min) - 1
    
    return _noise
    
def simplex_noise(length, sample_rate=44100, octaves=1, persistence=0.5, lacunarity=2.0, frequency=1000.0):
    """
    Generate Simplex noise using snoise2 with one dimension fixed.
    Parameters:
    - frequency: Controls the perceived frequency of the noise (higher = faster changes).
    Requires: pip install noise
    """
    import noise  # Lazy import for optional dependency
    num_samples = int(length * sample_rate)
    _noise = np.zeros(num_samples)

    # Use snoise2 with y fixed at 0 to simulate 1D Simplex noise
    for i in range(num_samples):
        _noise[i] = noise.snoise2(i * frequency / sample_rate, 0, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
    
    # Normalize to [-1, 1]
    _min, _max = np.min(_noise), np.max(_noise)
    if _max - _min > 0:
        _noise = 2 * (_noise - _min) / (_max - _min) - 1
    
    return _noise

def fractal_noise(length, sample_rate=44100, base_noise_generator=perlin_noise, octaves=4):
    """Generate fractal noise by summing multiple octaves of base noise."""
    num_samples = int(length * sample_rate)
    noise = np.zeros(num_samples)
    frequency = 1.0
    amplitude = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):
        noise += base_noise_generator(length, sample_rate) * amplitude
        max_amplitude += amplitude
        amplitude *= 0.5
        frequency *= 2.0

    return noise / max_amplitude  # Normalize to [-1, 1]

def granular_noise(length, sample_rate=44100, grain_size=0.05, overlap=0.5):
    """
    Generate granular noise.
    Parameters:
    - grain_size: Duration of each grain in seconds.
    - overlap: Fraction of overlap between consecutive grains.
    """
    num_samples = int(length * sample_rate)
    grain_samples = int(grain_size * sample_rate)
    stride = int(grain_samples * (1 - overlap))
    signal = np.zeros(num_samples)

    for start in range(0, num_samples, stride):
        end = min(start + grain_samples, num_samples)
        grain = np.random.uniform(-1, 1, grain_samples)
        signal[start:end] += grain[:end - start]

    return signal / max(np.abs(signal))  # Normalize to [-1, 1]

def random_walk_noise(length, sample_rate=44100, step_size=0.05):
    """
    Generate random walk noise.
    Parameters:
    - step_size: Maximum step size for each random walk step.
    """
    num_samples = int(length * sample_rate)
    signal = np.zeros(num_samples)
    for i in range(1, num_samples):
        step = np.random.uniform(-step_size, step_size)
        signal[i] = signal[i - 1] + step
    return signal / max(np.abs(signal))  # Normalize to [-1, 1]

def chaotic_noise(length, sample_rate=44100, r=3.9, x0=0.5):
    """
    Generate chaotic noise using the logistic map.
    Parameters:
    - r: Control parameter for chaos (typical range: 3.5 to 4.0).
    - x0: Initial condition.
    """
    num_samples = length * sample_rate
    signal = np.zeros(num_samples)
    x = x0
    for i in range(num_samples):
        x = r * x * (1 - x)
        signal[i] = 2 * (x - 0.5)  # Scale to [-1, 1]
    return signal

def markov_chain_noise(length, sample_rate=44100, states=(-1, 0, 1), transition_matrix=None):
    """
    Generate Markov Chain-based noise.
    Parameters:
    - states: Possible values for the noise.
    - transition_matrix: Square matrix defining state transition probabilities.
    """
    num_samples = int(length * sample_rate)
    if transition_matrix is None:
        # Default: Equal probabilities for transitions
        n = len(states)
        transition_matrix = np.ones((n, n)) / n

    state = np.random.choice(states)  # Start with a random state
    signal = []
    for _ in range(num_samples):
        signal.append(state)
        state = np.random.choice(states, p=transition_matrix[states.index(state)])

    return np.array(signal)

def tonal_noise(length, sample_rate=44100, frequency=440, noise_level=0.2):
    """
    Generate tonal noise.
    Parameters:
    - frequency: Frequency of the tonal component in Hz.
    - noise_level: Amplitude of the noise relative to the tone.
    """
    num_samples = int(length * sample_rate)
    t = np.linspace(0, length, num_samples, endpoint=False)
    tone = np.sin(2 * np.pi * frequency * t)
    noise = np.random.uniform(-1, 1, num_samples) * noise_level
    return (tone + noise) / max(abs(tone + noise))  # Normalize

def stochastic_resonance_noise(length, sample_rate=44100, weak_signal_frequency=2, noise_level=0.2):
    """
    Generate stochastic resonance noise.
    Parameters:
    - weak_signal_frequency: Frequency of the weak signal in Hz.
    - noise_level: Amplitude of the added noise.
    """
    num_samples = length * sample_rate
    t = np.linspace(0, length, num_samples, endpoint=False)
    weak_signal = np.sin(2 * np.pi * weak_signal_frequency * t) * 0.1  # Weak signal
    noise = np.random.uniform(-1, 1, num_samples) * noise_level
    return (weak_signal + noise) / max(abs(weak_signal + noise))  # Normalize

def cellular_automata_noise(length, sample_rate=44100, grid_size=100, rule=30):
    """
    Generate cellular automata-based noise.
    Parameters:
    - grid_size: Size of the 1D cellular automaton.
    - rule: Rule for the cellular automaton (e.g., Conway's Game of Life).
    """
    num_samples = int(length * sample_rate)
    generations = num_samples // grid_size
    grid = np.zeros((generations, grid_size), dtype=np.int8)
    grid[0, grid_size // 2] = 1  # Initial condition: single live cell in the center

    # Generate the cellular automaton
    for i in range(1, generations):
        for j in range(grid_size):
            neighborhood = (
                (grid[i - 1, (j - 1) % grid_size] << 2)
                | (grid[i - 1, j] << 1)
                | grid[i - 1, (j + 1) % grid_size]
            )
            grid[i, j] = (rule >> neighborhood) & 1

    # Convert grid to audio signal
    signal = grid.flatten()
    return signal / max(abs(signal))  # Normalize to [-1, 1]

