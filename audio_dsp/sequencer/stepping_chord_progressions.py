"""
Microtonal chord progression generator.

Generates chord progressions using microtonal scales (e.g., 24-TET quarter tones)
with stepping/walking patterns based on numeric sequences.

This module provides both the legacy functional API and the new unified
class-based API via ChordProgressionSequencer.
"""

import numpy as np
import time

try:
    from audio_dsp.utils import wav_io as wavfile
except ImportError:
    wavfile = None

# Import unified base classes
try:
    from .base import GenerativeSequencer
    _HAS_UNIFIED_API = True
except ImportError:
    _HAS_UNIFIED_API = False


# Audio parameters
SAMPLE_RATE = 44100
FADE_DURATION = 0.02

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Microtonal modes (steps in custom tuning, e.g., 24-TET)
MICROTONAL_MODES = {
    'micro_ionian': [4, 3, 2, 4, 4, 4, 2],
    'micro_dorian': [4, 2, 4, 4, 4, 2, 4],
    'micro_phrygian': [2, 4, 4, 4, 2, 4, 4],
    'micro_lydian': [4, 4, 4, 2, 4, 4, 2],
    'micro_mixolydian': [4, 4, 2, 4, 4, 2, 4],
    'micro_aeolian': [4, 2, 4, 4, 2, 4, 4],
    'micro_locrian': [2, 4, 4, 2, 4, 4, 4],
    'micro_harmonic_minor': [4, 2, 4, 4, 2, 6, 2],
    'micro_melodic_minor': [4, 2, 4, 4, 4, 4, 2],
    'phrygian_dominant': [2, 6, 2, 4, 2, 4, 4],
    'quarter_tone': [3, 3, 3, 3, 3, 3, 3, 3],
    'micro_blues': [6, 4, 2, 2, 6, 4]
}


# =============================================================================
# Unified API (new)
# =============================================================================

if _HAS_UNIFIED_API:
    class ChordProgressionSequencer(GenerativeSequencer):
        """
        Microtonal chord progression sequencer.

        Generates chord progressions using customizable microtonal scales
        with stepping patterns.

        Parameters
        ----------
        sample_rate : int
            Audio sample rate (default: 44100)
        bpm : float
            Tempo in BPM (default: 120)
        root_note : str
            Root note name (default: 'C')
        root_octave : int
            Root octave (default: 4)
        steps_per_octave : int
            Steps per octave for microtonal tuning (default: 24)

        Examples
        --------
        >>> seq = ChordProgressionSequencer(bpm=90, root_note='A', root_octave=3)
        >>> seq.set_mode('micro_dorian')
        >>> audio = seq.generate_stepped_progression(
        ...     start_num=1, step_sizes=[2, 3], num_steps=16, notes_per_chord=4
        ... )
        """

        MODES = MICROTONAL_MODES
        NOTE_NAMES = NOTE_NAMES

        def __init__(self, sample_rate: int = 44100, bpm: float = 120,
                     root_note: str = 'C', root_octave: int = 4,
                     steps_per_octave: int = 24):
            super().__init__(sample_rate, bpm)
            self.root_note = root_note.upper()
            self.root_octave = root_octave
            self.steps_per_octave = steps_per_octave
            self._mode_steps = self.MODES['micro_ionian']
            self._scale = None

        @property
        def root_freq(self) -> float:
            """Calculate root frequency from note and octave."""
            root_idx = self.NOTE_NAMES.index(self.root_note)
            return 440.0 * (2 ** ((root_idx - 9) / 12)) * (2 ** (self.root_octave - 4))

        def set_mode(self, mode_name: str = None, custom_steps: list = None):
            """
            Set the scale mode.

            Parameters
            ----------
            mode_name : str, optional
                Preset mode name from MICROTONAL_MODES
            custom_steps : list, optional
                Custom step pattern for the scale
            """
            if custom_steps:
                self._mode_steps = custom_steps
            elif mode_name and mode_name in self.MODES:
                self._mode_steps = self.MODES[mode_name]
            else:
                raise ValueError(f"Invalid mode: {mode_name}. Available: {list(self.MODES.keys())}")
            self._scale = None  # Reset cached scale

        def generate_scale(self) -> list:
            """Generate microtonal scale frequencies."""
            step_size = 2 ** (1 / self.steps_per_octave)
            scale = [self.root_freq]
            total_steps = 0
            for step in self._mode_steps[:-1]:
                total_steps += step
                freq = self.root_freq * (step_size ** total_steps)
                scale.append(freq)
            self._scale = scale
            return scale

        def reduce_to_degree(self, number: int) -> int:
            """Reduce a number to a scale degree (1-7)."""
            total = number
            while total > 9:
                total = sum(int(digit) for digit in str(total))
            result = total % 7
            return result if result > 0 else 7

        def build_chord(self, degree: int, notes_per_chord: int = 3) -> list:
            """Build a chord from a scale degree."""
            if self._scale is None:
                self.generate_scale()

            degree = degree - 1  # 0-indexed
            chord = []
            for i in range(notes_per_chord):
                index = (degree + i * 2) % len(self._scale)
                chord.append(self._scale[index])
            return chord

        def generate(self, duration: float) -> np.ndarray:
            """Generate chord progression for specified duration."""
            # Default stepping pattern
            return self.generate_stepped_progression(
                start_num=1,
                step_sizes=[2, 3, 5],
                num_steps=int(duration / self.beat_duration),
                notes_per_chord=3
            )

        def generate_stepped_progression(self, start_num: int = 1,
                                         step_sizes: list = [2, 3],
                                         num_steps: int = 16,
                                         notes_per_chord: int = 3) -> np.ndarray:
            """
            Generate a stepped chord progression.

            Parameters
            ----------
            start_num : int
                Starting number for stepping (default: 1)
            step_sizes : list
                Step sizes to cycle through (default: [2, 3])
            num_steps : int
                Number of chords to generate (default: 16)
            notes_per_chord : int
                Notes per chord (3=triad, 4=seventh, etc.)

            Returns
            -------
            np.ndarray
                Generated audio
            """
            # Generate progression
            progression = []
            current_num = start_num
            for i in range(num_steps):
                step = step_sizes[i % len(step_sizes)]
                degree = self.reduce_to_degree(current_num)
                chord = self.build_chord(degree, notes_per_chord)
                progression.append((degree, chord))
                current_num += step

            return self._generate_progression_audio(progression)

        def _generate_progression_audio(self, progression: list) -> np.ndarray:
            """Generate audio from a progression list."""
            samples_per_chord = int(self.beat_duration * self.sample_rate)
            total_samples = samples_per_chord * len(progression)
            output = np.zeros(total_samples, dtype=np.float32)

            t = np.linspace(0, self.beat_duration, samples_per_chord, endpoint=False)
            fade_samples = int(self.sample_rate * FADE_DURATION)
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)

            for i, (degree, chord) in enumerate(progression):
                start_idx = i * samples_per_chord
                chord_wave = np.zeros(samples_per_chord, dtype=np.float32)

                for freq in chord:
                    chord_wave += np.sin(2 * np.pi * freq * t) * 0.3
                chord_wave /= len(chord)

                if fade_samples < samples_per_chord:
                    chord_wave[:fade_samples] *= fade_in
                    chord_wave[-fade_samples:] *= fade_out

                output[start_idx:start_idx + samples_per_chord] = chord_wave

            return self._normalize(output)


# =============================================================================
# Legacy API (backwards compatible)
# =============================================================================

def generate_scale(root_freq, steps_per_octave, mode_steps):
    """Generate microtonal scale."""
    print("Generating microtonal scale")
    scale = [root_freq]
    step_size = 2 ** (1 / steps_per_octave)
    total_steps = 0
    for step in mode_steps[:-1]:
        total_steps += step
        current_freq = root_freq * (step_size ** total_steps)
        scale.append(current_freq)
    print(f"Scale frequencies: {[f'{f:.2f}' for f in scale]}")
    return scale


def reduce_to_degree(number):
    """Reduce number to scale degree."""
    total = number
    while total > 9:
        total = sum(int(digit) for digit in str(total))
    result = total % 7
    return result if result > 0 else 7


def build_chord(degree, scale, notes_per_chord):
    """Build chord from scale degree."""
    degree = degree - 1
    chord = []
    scale_len = len(scale)
    for i in range(notes_per_chord):
        index = (degree + i * 2) % scale_len
        chord.append(scale[index])
    return chord


def generate_progression(start_num, step_sizes, num_steps, scale, notes_per_chord):
    """Generate stepped chord progression."""
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
    """Generate audio from progression."""
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
        if wavfile:
            wavfile.write("microtonal_progression.wav", SAMPLE_RATE, audio_data)
        print(f"File write time: {time.time() - start_time:.2f} seconds")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
