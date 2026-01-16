import numpy as np
import pyaudio
import time
from sympy import Matrix
from audio_dsp.utils import wav_io as wavfile

# Initialize PyAudio
p = pyaudio.PyAudio()

# Audio parameters
SAMPLE_RATE = 44100  # Hz
BASE_DURATION = 0.3  # Base note duration in seconds (will be scaled by BPM)

# Pentatonic scale frequencies (C pentatonic: C, D, E, G, A)
PENTATONIC_BASE = [261.63, 293.66, 329.63, 392.00, 440.00]  # Hz (C4 scale)

# Function to generate a sine wave (returns samples instead of playing directly)
def generate_tone(frequency, duration, volume=0.5):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    tone = np.sin(frequency * t * 2 * np.pi) * volume
    return (tone * 32767).astype(np.int16)

# Play tone live and append to audio buffer
def play_and_buffer_tone(frequency, duration, volume=0.5, stream=None, audio_buffer=None):
    samples = generate_tone(frequency, duration, volume)
    if stream is None:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, output=True)
    stream.write(samples.tobytes())
    if audio_buffer is not None:
        audio_buffer.extend(samples)
    return stream

# Map values to pentatonic scale
def map_to_pentatonic(values, row_wise=False):
    if not row_wise:
        values_flat = values.flatten()
        min_val = np.min(values_flat)
        max_val = np.max(values_flat)
        if max_val == min_val:
            return np.array([PENTATONIC_BASE[0]] * values.size).reshape(values.shape)
        normalized = (values_flat - min_val) / (max_val - min_val)
        indices = np.round(normalized * (len(PENTATONIC_BASE) - 1)).astype(int)
        return np.array([PENTATONIC_BASE[i] for i in indices]).reshape(values.shape)
    else:
        freqs = []
        for row in values:
            min_val = np.min(row)
            max_val = np.max(row)
            if max_val == min_val:
                freqs.append([PENTATONIC_BASE[0]] * len(row))
            else:
                normalized = (row - min_val) / (max_val - min_val)
                indices = np.round(normalized * (len(PENTATONIC_BASE) - 1)).astype(int)
                freqs.append([PENTATONIC_BASE[i] for i in indices])
        return np.array(freqs)

# Play and buffer a matrix as arpeggiated chords
def play_matrix(matrix, label, tempo, audio_buffer):
    print(f"\n{label}:")
    print(matrix)
    freqs = map_to_pentatonic(matrix, row_wise=True)
    duration = 60 / tempo  # Duration per beat (in seconds)
    print(f"\nPlaying {label} (BPM: {tempo})...")
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, output=True)
    for row in freqs:
        for freq in row:
            play_and_buffer_tone(freq, duration * BASE_DURATION, stream=stream, audio_buffer=audio_buffer)
        time.sleep(duration * 0.1)  # Short pause between rows
    stream.stop_stream()
    stream.close()

# Play and buffer a matrix as a single chord
def play_chord(matrix, label, duration, audio_buffer):
    print(f"\n{label}:")
    print(matrix)
    freqs = map_to_pentatonic(matrix, row_wise=True)
    chord_freqs = [row[0] for row in freqs[:4]]  # First note from each row
    print(f"Playing {label} as chord: {chord_freqs}...")
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, output=True)
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    combined_wave = np.zeros_like(t)
    for freq in chord_freqs:
        combined_wave += np.sin(freq * t * 2 * np.pi) * 0.3
    samples = (combined_wave * 32767 / len(chord_freqs)).astype(np.int16)
    stream.write(samples.tobytes())
    audio_buffer.extend(samples)
    stream.stop_stream()
    stream.close()

# Play and buffer eigenvalues as a sustained chord
def play_eigenvalues(eigenvalues, duration, audio_buffer):
    eig_magnitudes = np.abs(eigenvalues)
    freqs = map_to_pentatonic(eig_magnitudes)
    print(f"\nEigenvalues (magnitudes): {eig_magnitudes}")
    print(f"Playing eigenvalues as chord: {freqs}...")
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, output=True)
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    combined_wave = np.zeros_like(t)
    for freq in freqs:
        combined_wave += np.sin(freq * t * 2 * np.pi) * 0.3
    samples = (combined_wave * 32767 / len(freqs)).astype(np.int16)
    stream.write(samples.tobytes())
    audio_buffer.extend(samples)
    stream.stop_stream()
    stream.close()

# Manual RREF with step-by-step chord playback
def play_rref_steps(matrix, tempo, audio_buffer):
    print("\nRREF Transformation Steps:")
    m = matrix.copy().astype(float)
    rows, cols = m.shape
    steps = [m.copy()]
    lead = 0

    for r in range(rows):
        if lead >= cols:
            break
        pivot = m[r, lead]
        i = r
        while pivot == 0 and i < rows:
            i += 1
            if i == rows:
                i = r
                lead += 1
                if lead == cols:
                    break
                pivot = m[i, lead]
            else:
                m[[i, r]] = m[[r, i]]
                steps.append(m.copy())
                pivot = m[r, lead]

        if lead < cols:
            if pivot != 0:
                m[r] = m[r] / pivot
                steps.append(m.copy())
            for i in range(rows):
                if i != r:
                    factor = m[i, lead]
                    m[i] -= factor * m[r]
                    steps.append(m.copy())
            lead += 1

    duration = 60 / tempo * 2 * BASE_DURATION  # Chord duration based on BPM
    for i, step_matrix in enumerate(steps):
        play_chord(step_matrix, f"RREF Step {i}", duration, audio_buffer)
        time.sleep(duration * 0.1)  # Slight pause between steps

    return m

# Generate a random square matrix
def generate_random_matrix(size, min_val, max_val):
    return np.random.uniform(min_val, max_val, (size, size))

# Get matrix input from user or generate randomly
def get_matrix():
    choice = input("Enter 'r' for random matrix or 'm' for manual input: ").lower()
    if choice == 'r':
        size = int(input("Enter matrix size (e.g., 3 for 3x3): "))
        min_val = float(input("Enter minimum value: "))
        max_val = float(input("Enter maximum value: "))
        return generate_random_matrix(size, min_val, max_val)
    else:
        print("Enter a square matrix. Type each row as space-separated numbers.")
        print("Example: '1 2 3' for one row. Press Enter after each row, type 'done' when finished.")
        matrix = []
        while True:
            row_input = input("Enter row (or 'done'): ")
            if row_input.lower() == 'done':
                break
            try:
                row = [float(x) for x in row_input.split()]
                matrix.append(row)
            except ValueError:
                print("Invalid input. Use numbers separated by spaces.")
        matrix = np.array(matrix)
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square (e.g., 4x4).")
        return matrix

# Main execution
try:
    # Get BPM from user
    bpm = float(input("Enter tempo in BPM (e.g., 120): "))
    
    # Initialize audio buffer for WAV output
    audio_buffer = []

    # Get matrix
    matrix = get_matrix()
    det = np.linalg.det(matrix)
    print(f"\nDeterminant: {det}")

    # Play and buffer components
    play_matrix(matrix, "Original Matrix", bpm, audio_buffer)
    eigenvalues = np.linalg.eigvals(matrix)
    play_eigenvalues(eigenvalues, 2.0, audio_buffer)
    rref_matrix = play_rref_steps(matrix, bpm, audio_buffer)

    # Save to WAV file
    wavfile.write("matrix_music.wav", SAMPLE_RATE, np.array(audio_buffer))
    print("\nAudio saved to 'matrix_music.wav'")

except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Clean up
p.terminate()