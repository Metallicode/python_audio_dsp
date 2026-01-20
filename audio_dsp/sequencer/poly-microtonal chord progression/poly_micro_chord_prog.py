"""
Poly-microtonal chord progression generator with sample pitch shifting.

Generates chord progressions using microtonal tunings (various EDO systems)
by pitch-shifting a source sample. Supports multiple tuning systems in a
single progression.

This module provides both the legacy functional API and the new unified
class-based API via PolyMicrotonalSequencer.
"""

import numpy as np
import re

try:
    from audio_dsp.utils import wav_io as wavfile
    from audio_dsp.utils import load_audio, resample_audio
except ImportError:
    wavfile = None
    load_audio = None
    resample_audio = None

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

# Import unified base classes
try:
    from ..base import PatternBasedSequencer
    from ..sample_manager import SampleManager
    _HAS_UNIFIED_API = True
except ImportError:
    _HAS_UNIFIED_API = False


SAMPLE_RATE = 44100


# =============================================================================
# Unified API (new)
# =============================================================================

if _HAS_UNIFIED_API:
    class PolyMicrotonalSequencer(PatternBasedSequencer):
        """
        Poly-microtonal chord progression sequencer with sample pitch shifting.

        Uses a source sample and pitch-shifts it to create chords in various
        microtonal tuning systems (EDO - Equal Divisions of the Octave).

        Parameters
        ----------
        sample_rate : int
            Audio sample rate (default: 44100)
        bpm : float
            Tempo in BPM (default: 120)
        voices : int
            Number of voices per chord (default: 3)

        Examples
        --------
        >>> seq = PolyMicrotonalSequencer(bpm=60)
        >>> seq.load_source_sample("sample.wav")
        >>> chord_str = "Cm-I-31-1/4, G-IV-19-1/4"
        >>> audio = seq.generate_from_chord_string(chord_str)
        """

        DEGREE_MAP = {'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5, 'VII': 6}
        NOTE_MAP = {'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4,
                    'F#': -3, 'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2}

        def __init__(self, sample_rate: int = 44100, bpm: float = 120, voices: int = 3):
            super().__init__(sample_rate, bpm)
            self.voices = voices
            self._source_sample = None

        def load_source_sample(self, sample_path_or_array, base_freq: float = 440.0):
            """
            Load the source sample for pitch shifting.

            Parameters
            ----------
            sample_path_or_array : str or np.ndarray
                File path or numpy array
            base_freq : float
                Base frequency of the source sample (default: 440.0)
            """
            self.base_freq = base_freq

            if isinstance(sample_path_or_array, np.ndarray):
                self.sample_manager.add("source", sample_path_or_array, self.sample_rate)
            else:
                self.sample_manager.load(sample_path_or_array, "source")

            self._source_sample = self.sample_manager.get("source")

        def pitch_shift(self, target_freq: float) -> np.ndarray:
            """
            Pitch shift the source sample to target frequency.

            Parameters
            ----------
            target_freq : float
                Target frequency in Hz

            Returns
            -------
            np.ndarray
                Pitch-shifted audio
            """
            if self._source_sample is None:
                raise ValueError("No source sample loaded. Call load_source_sample() first.")

            if not _HAS_LIBROSA:
                raise ImportError("librosa is required for pitch shifting")

            rate = target_freq / self.base_freq
            n_steps = np.log2(rate) * 12
            return librosa.effects.pitch_shift(self._source_sample, sr=self.sample_rate, n_steps=n_steps)

        def get_chord_steps(self, quality: str) -> list:
            """Get chord intervals based on quality."""
            steps = {
                "major": [0, 4, 7],
                "m": [0, 3, 7],
                "sus": [0, 5, 7],
                "7": [0, 4, 7, 10]
            }
            return steps.get(quality, steps["major"])[:self.voices]

        def parse_chord_string(self, chord_str: str) -> list:
            """
            Parse chord string into structured chord data.

            Parameters
            ----------
            chord_str : str
                Chord string like "Cm-I-31-1/4, G-IV-16-1/4"

            Returns
            -------
            list
                List of (note, quality, degree, edo, duration) tuples
            """
            chords = chord_str.split(", ")
            pattern = re.compile(r"([A-G]#?)(m|sus|7)?-([IV]+)-(\d+)-(\d+/\d+)")
            result = []

            for chord in chords:
                match = pattern.match(chord.strip())
                if match:
                    note, quality, degree_str, edo, dur_str = match.groups()
                    duration = eval(dur_str)
                    quality = quality if quality else "major"
                    if degree_str not in self.DEGREE_MAP:
                        raise ValueError(f"Invalid degree '{degree_str}'")
                    degree = self.DEGREE_MAP[degree_str]
                    result.append((note, quality, degree, int(edo), duration))

            return result

        def generate(self, duration: float) -> np.ndarray:
            """Generate audio for specified duration."""
            return np.zeros(int(duration * self.sample_rate), dtype=np.float32)

        def generate_from_chord_string(self, chord_str: str) -> np.ndarray:
            """
            Generate audio from a chord string.

            Parameters
            ----------
            chord_str : str
                Chord progression string

            Returns
            -------
            np.ndarray
                Generated audio
            """
            chords = self.parse_chord_string(chord_str)
            seconds_per_beat = 60 / self.bpm
            total_duration = sum(dur * seconds_per_beat for _, _, _, _, dur in chords)
            total_samples = int(total_duration * self.sample_rate)
            output = np.zeros(total_samples, dtype=np.float32)

            current_sample = 0
            for note, quality, degree, n_edo, duration in chords:
                dur_sec = duration * seconds_per_beat
                dur_samples = int(dur_sec * self.sample_rate)

                # Calculate frequencies
                steps_12edo = self.NOTE_MAP[note]
                base_freq = 440 * (2 ** (steps_12edo / 12))
                steps_from_a4 = n_edo * np.log2(base_freq / 440)

                root_steps = steps_from_a4 + degree
                chord_steps = self.get_chord_steps(quality)
                chord_audio = np.zeros(dur_samples, dtype=np.float32)

                for step in chord_steps:
                    total_steps = root_steps + (step * n_edo / 12)
                    target_freq = 440 * (2 ** (total_steps / n_edo))
                    pitched = self.pitch_shift(target_freq)

                    if len(pitched) > dur_samples:
                        pitched = pitched[:dur_samples]
                    elif len(pitched) < dur_samples:
                        pitched = np.pad(pitched, (0, dur_samples - len(pitched)), 'constant')

                    chord_audio += pitched / len(chord_steps)

                end_sample = min(current_sample + dur_samples, total_samples)
                output[current_sample:end_sample] += chord_audio[:end_sample - current_sample]
                current_sample += dur_samples

            return self._normalize(output)


# =============================================================================
# Legacy API (backwards compatible)
# =============================================================================

def save_wav(file_path, data):
    """Save audio to WAV file."""
    data = np.clip(data, -1, 1)
    if wavfile:
        wavfile.write(file_path, SAMPLE_RATE, (data * 32767).astype(np.int16))
    print(f"Saved to {file_path}")


def pitch_shift(sample, sr, target_freq, base_freq=440.0):
    """Pitch shift sample to target frequency."""
    if not _HAS_LIBROSA:
        raise ImportError("librosa is required for pitch shifting")
    rate = target_freq / base_freq
    shifted = librosa.effects.pitch_shift(sample, sr=sr, n_steps=np.log2(rate) * 12)
    return shifted


def parse_chord_string(chord_str):
    """Parse chord string into structured data."""
    chords = chord_str.split(", ")
    pattern = re.compile(r"([A-G]#?)(m|sus|7)?-([IV]+)-(\d+)-(\d+/\d+)")
    degree_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5, 'VII': 6}
    result = []
    for chord in chords:
        match = pattern.match(chord.strip())
        if match:
            note, quality, degree_str, edo, dur_str = match.groups()
            duration = eval(dur_str)
            quality = quality if quality else "major"
            if degree_str not in degree_map:
                raise ValueError(f"Invalid degree '{degree_str}' in '{chord}'.")
            degree = degree_map[degree_str]
            result.append((note, quality, degree, int(edo), duration))
        else:
            print(f"Invalid chord format: {chord}")
    return result


def note_to_steps(note, reference="A4"):
    """Convert note to steps from A4 in 12-EDO."""
    note_map = {'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4, 'F#': -3,
                'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2}
    return note_map[note]


def get_chord_steps(quality, voices):
    """Get chord step intervals."""
    if quality == "major":
        return [0, 4, 7][:voices]
    elif quality == "m":
        return [0, 3, 7][:voices]
    elif quality == "sus":
        return [0, 5, 7][:voices]
    elif quality == "7":
        return [0, 4, 7, 10][:voices]
    else:
        raise ValueError(f"Unknown chord quality: {quality}")


def generate_chord_progression(chord_str, bpm, sample_file="sample.wav", voices=3):
    """Generate chord progression audio."""
    if load_audio is None:
        raise ImportError("audio_dsp.utils required for load_audio")

    sr, sample = load_audio(sample_file, mono=True)
    if sr != SAMPLE_RATE:
        sample = resample_audio(sample, sr, SAMPLE_RATE)

    chords = parse_chord_string(chord_str)
    print(f"Parsed chords: {chords}")

    seconds_per_beat = 60 / bpm
    total_duration = sum(dur * seconds_per_beat for _, _, _, _, dur in chords)
    total_samples = int(total_duration * SAMPLE_RATE)
    output = np.zeros(total_samples, dtype=np.float32)
    print(f"Total duration: {total_duration:.3f}s, total samples: {total_samples}")

    current_sample = 0
    for note, quality, degree, n_edo, duration in chords:
        dur_sec = duration * seconds_per_beat
        dur_samples = int(dur_sec * SAMPLE_RATE)
        chord_name = f"{note}{quality if quality != 'major' else ''}-{['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'][degree]}-{n_edo}-{duration}"
        print(f"Chord {chord_name}: {dur_sec:.3f}s, {dur_samples} samples")

        steps_12edo = note_to_steps(note)
        base_freq = 440 * (2 ** (steps_12edo / 12))
        steps_from_a4 = n_edo * np.log2(base_freq / 440)

        root_steps = steps_from_a4 + degree
        chord_steps = get_chord_steps(quality, voices)
        chord_audio = np.zeros(dur_samples, dtype=np.float32)

        for step in chord_steps:
            total_steps = root_steps + (step * n_edo / 12)
            target_freq = 440 * (2 ** (total_steps / n_edo))
            print(f"  Pitching {note} (degree {degree}, step {step}) to {target_freq:.2f} Hz in {n_edo}-EDO")
            pitched = pitch_shift(sample, SAMPLE_RATE, target_freq)
            if len(pitched) > dur_samples:
                pitched = pitched[:dur_samples]
            elif len(pitched) < dur_samples:
                pitched = np.pad(pitched, (0, dur_samples - len(pitched)), 'constant')
            chord_audio += pitched / len(chord_steps)

        end_sample = min(current_sample + dur_samples, total_samples)
        print(f"  Placing at {current_sample} to {end_sample}")
        output[current_sample:end_sample] += chord_audio[:end_sample - current_sample]
        current_sample += dur_samples

    return output


def main(chord_str, bpm, output_file="progression_with_degrees.wav"):
    """Generate and save chord progression."""
    print(f"Generating progression at {bpm} BPM...")
    audio = generate_chord_progression(chord_str, bpm)
    save_wav(output_file, audio)


if __name__ == "__main__":
    chord_str = (
        # A1 (Bars 1-8): Moody opening with new EDOs
        "Cm-I-31-1/4, G-IV-19-1/4, A#-VII-12-1/4, F7-II-24-1/4, "    # 1-4
        "Cm-III-43-1/4, Gsus-I-22-1/4, D-V-53-1/4, F-IV-19-1/4, "     # 5-8

        # A2 (Bars 9-16): Variation, building tension
        "Cm-I-31-1/4, G7-VI-22-1/4, A#-III-12-1/4, F-II-43-1/4, "     # 9-12
        "Cm-V-53-1/4, D-II-24-1/4, Gsus-IV-19-1/4, F7-VII-31-1/4, "   # 13-16

        # B (Bars 17-24): Bright, exotic contrast
        "E-I-22-1/4, B7-IV-43-1/4, F#-VI-12-1/4, C-III-53-1/4, "      # 17-20
        "E-V-19-1/4, Bsus-II-31-1/4, G-VII-24-1/4, D7-I-22-1/4, "     # 21-24

        # A3 (Bars 25-32): Return and resolve
        "Cm-I-31-1/4, G-III-19-1/4, A#-V-12-1/4, F-IV-43-1/4, "       # 25-28
        "Cm-VI-53-1/4, Gsus-I-22-1/4, D-IV-24-1/4, F7-II-31-1/4"      # 29-32
    )
    bpm = 10
    main(chord_str, bpm)
