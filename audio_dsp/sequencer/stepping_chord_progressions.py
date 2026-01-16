# import numpy as np
# from audio_dsp.utils import wav_io as wavfile
# import time

# print("Script started")

# # Audio parameters
# SAMPLE_RATE = 44100  # Hz
# FADE_DURATION = 0.02  # 20 ms fade in/out

# NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
# MODES = {
#     'ionian': [2, 2, 1, 2, 2, 2, 1],
#     'dorian': [2, 1, 2, 2, 2, 1, 2],
#     'phrygian': [1, 2, 2, 2, 1, 2, 2],
#     'lydian': [2, 2, 2, 1, 2, 2, 1],
#     'mixolydian': [2, 2, 1, 2, 2, 1, 2],
#     'aeolian': [2, 1, 2, 2, 1, 2, 2],
#     'locrian': [1, 2, 2, 1, 2, 2, 2],
#     'harmonic_minor': [2, 1, 2, 2, 1, 3, 1],
#     'melodic_minor': [2, 1, 2, 2, 2, 2, 1],
#     'phrygian_dominant': [1, 3, 1, 2, 1, 2, 2],
#     'whole_tone': [2, 2, 2, 2, 2, 2],          # 6 notes
#     'blues': [3, 2, 1, 1, 3, 2],              # 6 notes
#     'pentatonic_major': [2, 2, 3, 2, 3],      # 5 notes
#     'pentatonic_minor': [3, 2, 2, 3, 2]       # 5 notes
# }

# def generate_scale(root_freq, mode):
#     print("Generating scale")
#     intervals = MODES[mode]
#     scale = [root_freq]
#     current_freq = root_freq
#     for interval in intervals[:-1]:
#         current_freq *= 2 ** (interval / 12)
#         scale.append(current_freq)
#     return scale

# def reduce_to_degree(number):
#     total = number
#     while total > 9:
#         total = sum(int(digit) for digit in str(total))
#     result = total % 7
#     return result if result > 0 else 7

# def build_chord(degree, scale, notes_per_chord):
#     degree = degree - 1
#     chord = []
#     for i in range(notes_per_chord):
#         index = (degree + i * 2) % len(scale)
#         chord.append(scale[index])
#     return chord

# def generate_progression(start_num, step_sizes, num_steps, scale, notes_per_chord):
#     print("Generating progression")
#     progression = []
#     current_num = start_num
#     for i in range(num_steps):
#         step = step_sizes[i % len(step_sizes)]  # Cycle through step sizes
#         print(f"Iteration {i+1}, current_num = {current_num}, step = {step}")
#         degree = reduce_to_degree(current_num)
#         chord = build_chord(degree, scale, notes_per_chord)
#         progression.append((degree, chord))
#         current_num += step
#     return progression

# def generate_full_audio(progression, bpm):
#     print("Starting audio generation")
#     duration = 60 / bpm
#     samples_per_chord = int(SAMPLE_RATE * duration)
#     total_samples = samples_per_chord * len(progression)
#     audio_data = np.zeros(total_samples, dtype=np.float32)
#     t = np.linspace(0, duration, samples_per_chord, False)
    
#     fade_samples = int(SAMPLE_RATE * FADE_DURATION)
#     fade_in = np.linspace(0, 1, fade_samples)
#     fade_out = np.linspace(1, 0, fade_samples)
    
#     for i, (degree, chord) in enumerate(progression):
#         print(f"Step {i+1}: Degree {degree}")
#         start_idx = i * samples_per_chord
#         chord_wave = np.zeros(samples_per_chord, dtype=np.float32)
#         for freq in chord:
#             chord_wave += np.sin(freq * t * 2 * np.pi) * 0.3
#         chord_wave /= len(chord)
        
#         if fade_samples < samples_per_chord:
#             chord_wave[:fade_samples] *= fade_in
#             chord_wave[-fade_samples:] *= fade_out
        
#         audio_data[start_idx:start_idx + samples_per_chord] = chord_wave
    
#     return (audio_data * 32767).astype(np.int16)

# try:
#     print("Getting inputs")
#     bpm = float(input("Enter tempo in BPM (e.g., 120): "))
#     root_note = input("Enter root note (e.g., C, G#, A): ").upper()
#     root_octave = int(input("Enter root octave (e.g., 4 for C4): "))
#     mode = input("Enter mode (ionian, dorian, phrygian, lydian, mixolydian, aeolian, locrian): ").lower()
#     notes_per_chord = int(input("Enter number of notes per chord (e.g., 3 for triad, 4 for seventh): "))
#     start_num = int(input("Enter starting number (e.g., 1): "))
#     step_input = input("Enter step sizes as space-separated numbers (e.g., 2 45 3): ")
#     step_sizes = [int(x) for x in step_input.split()]
#     num_steps = int(input("Enter number of steps (e.g., 40): "))

#     print("Validating inputs")
#     if not step_sizes:
#         raise ValueError("Step sizes list cannot be empty")
#     if root_note not in NOTE_NAMES or mode not in MODES or notes_per_chord < 1:
#         raise ValueError("Invalid input")

#     print("Calculating root frequency")
#     root_idx = NOTE_NAMES.index(root_note)
#     root_freq = 440.0 * (2 ** ((root_idx - 9) / 12)) * (2 ** (root_octave - 4))

#     start_time = time.time()
#     scale = generate_scale(root_freq, mode)
#     progression = generate_progression(start_num, step_sizes, num_steps, scale, notes_per_chord)
#     print(f"Setup time: {time.time() - start_time:.2f} seconds")

#     print("Generating audio")
#     start_time = time.time()
#     audio_data = generate_full_audio(progression, bpm)
#     print(f"Audio generation time: {time.time() - start_time:.2f} seconds")

#     print("Writing file")
#     start_time = time.time()
#     wavfile.write("chord_progression.wav", SAMPLE_RATE, audio_data)
#     print(f"File write time: {time.time() - start_time:.2f} seconds")

# except ValueError as e:
#     print(f"Error: {e}")
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")



import numpy as np
from audio_dsp.utils import wav_io as wavfile
import time

# Audio parameters
SAMPLE_RATE = 44100  # Hz
FADE_DURATION = 0.02  # 20 ms fade in/out

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Microtonal modes (steps in custom tuning, e.g., 24-TET)
MICROTONAL_MODES = {
    'micro_ionian': [4, 3, 2, 4, 4, 4, 2],      # ~Major in 24-TET
    'micro_dorian': [4, 2, 4, 4, 4, 2, 4],      # ~Dorian
    'micro_phrygian': [2, 4, 4, 4, 2, 4, 4],    # ~Phrygian
    'micro_lydian': [4, 4, 4, 2, 4, 4, 2],      # ~Lydian
    'micro_mixolydian': [4, 4, 2, 4, 4, 2, 4],  # ~Mixolydian
    'micro_aeolian': [4, 2, 4, 4, 2, 4, 4],     # ~Aeolian
    'micro_locrian': [2, 4, 4, 2, 4, 4, 4],     # ~Locrian
    'micro_harmonic_minor': [4, 2, 4, 4, 2, 6, 2],  # ~Harmonic Minor
    'micro_melodic_minor': [4, 2, 4, 4, 4, 4, 2],  # ~Melodic Minor
    'phrygian_dominant': [2, 6, 2, 4, 2, 4, 4],    # ~Phrygian Dominant
    'quarter_tone': [3, 3, 3, 3, 3, 3, 3, 3],      # Even quarter tones
    'micro_blues': [6, 4, 2, 2, 6, 4]              # ~Blues
}

def generate_scale(root_freq, steps_per_octave, mode_steps):
    print("Generating microtonal scale")
    scale = [root_freq]
    step_size = 2 ** (1 / steps_per_octave)
    total_steps = 0
    for step in mode_steps[:-1]:  # Stop before octave repeat
        total_steps += step
        current_freq = root_freq * (step_size ** total_steps)
        scale.append(current_freq)
    print(f"Scale frequencies: {[f'{f:.2f}' for f in scale]}")
    return scale

def reduce_to_degree(number):
    total = number
    while total > 9:
        total = sum(int(digit) for digit in str(total))
    result = total % 7
    return result if result > 0 else 7

def build_chord(degree, scale, notes_per_chord):
    degree = degree - 1
    chord = []
    scale_len = len(scale)
    for i in range(notes_per_chord):
        index = (degree + i * 2) % scale_len
        chord.append(scale[index])
    return chord

def generate_progression(start_num, step_sizes, num_steps, scale, notes_per_chord):
    print("Generating progression")
    progression = []
    current_num = start_num
    for i in range(num_steps):
        step = step_sizes[i % len(step_sizes)]
        print(f"Iteration {i+1}, current_num = {current_num}, step = {step}")
        degree = reduce_to_degree(current_num)
        chord = build_chord(degree, scale, notes_per_chord)
        progression.append((degree, chord))
        current_num += step
    return progression

def generate_full_audio(progression, bpm):
    print("Starting audio generation")
    duration = 60 / bpm
    samples_per_chord = int(SAMPLE_RATE * duration)
    total_samples = samples_per_chord * len(progression)
    audio_data = np.zeros(total_samples, dtype=np.float32)
    t = np.linspace(0, duration, samples_per_chord, False)
    
    fade_samples = int(SAMPLE_RATE * FADE_DURATION)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    for i, (degree, chord) in enumerate(progression):
        print(f"Step {i+1}: Degree {degree}")
        start_idx = i * samples_per_chord
        chord_wave = np.zeros(samples_per_chord, dtype=np.float32)
        for freq in chord:
            chord_wave += np.sin(freq * t * 2 * np.pi) * 0.3
        chord_wave /= len(chord)
        
        if fade_samples < samples_per_chord:
            chord_wave[:fade_samples] *= fade_in
            chord_wave[-fade_samples:] *= fade_out
        
        audio_data[start_idx:start_idx + samples_per_chord] = chord_wave
    
    return (audio_data * 32767).astype(np.int16)

if __name__ == "__main__":
    try:
        print("Getting inputs")
        bpm = float(input("Enter tempo in BPM (e.g., 120): "))
        root_note = input("Enter root note (e.g., C, G#, A): ").upper()
        root_octave = int(input("Enter root octave (e.g., 4 for C4): "))
        steps_per_octave = int(input("Enter steps per octave (e.g., 24 for quarter tones): "))

        mode_choice = input("Enter 'preset' for predefined mode or 'custom' for custom steps: ").lower()
        if mode_choice == 'preset':
            mode = input("Enter mode (micro_ionian, micro_dorian, micro_phrygian, micro_lydian, micro_mixolydian, micro_aeolian, micro_locrian, micro_harmonic_minor, micro_melodic_minor, phrygian_dominant, quarter_tone, micro_blues): ").lower()
            if mode not in MICROTONAL_MODES:
                raise ValueError("Invalid preset mode")
            mode_steps = MICROTONAL_MODES[mode]
        else:
            step_input = input("Enter mode steps as space-separated numbers (e.g., 4 2 4 4 4 2 4): ")
            mode_steps = [int(x) for x in step_input.split()]
            if not mode_steps:
                raise ValueError("Mode steps list cannot be empty")

        notes_per_chord = int(input("Enter number of notes per chord (e.g., 3 for triad, 4 for seventh): "))
        start_num = int(input("Enter starting number (e.g., 1): "))
        step_input = input("Enter step sizes as space-separated numbers (e.g., 2 45 3): ")
        step_sizes = [int(x) for x in step_input.split()]
        num_steps = int(input("Enter number of steps (e.g., 40): "))

        print("Validating inputs")
        if not step_sizes:
            raise ValueError("Step sizes list cannot be empty")
        if root_note not in NOTE_NAMES or (mode_choice == 'preset' and mode not in MICROTONAL_MODES) or notes_per_chord < 1:
            raise ValueError("Invalid input")

        print("Calculating root frequency")
        root_idx = NOTE_NAMES.index(root_note)
        root_freq = 440.0 * (2 ** ((root_idx - 9) / 12)) * (2 ** (root_octave - 4))

        start_time = time.time()
        scale = generate_scale(root_freq, steps_per_octave, mode_steps)
        progression = generate_progression(start_num, step_sizes, num_steps, scale, notes_per_chord)
        print(f"Setup time: {time.time() - start_time:.2f} seconds")

        print("Generating audio")
        start_time = time.time()
        audio_data = generate_full_audio(progression, bpm)
        print(f"Audio generation time: {time.time() - start_time:.2f} seconds")

        print("Writing file")
        start_time = time.time()
        wavfile.write("microtonal_progression.wav", SAMPLE_RATE, audio_data)
        print(f"File write time: {time.time() - start_time:.2f} seconds")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")