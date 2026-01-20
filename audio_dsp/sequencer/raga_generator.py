"""
Raga-based music generator with fractal rhythms.

Generates music using Indian raga scales with time-of-day appropriate
raga selection and fractal-based rhythm patterns.

This module provides both the legacy functional API and the new unified
class-based API via RagaSequencer.
"""

import numpy as np
from datetime import datetime, timedelta
import time
import random

try:
    import simpleaudio as sa
    _HAS_SIMPLEAUDIO = True
except ImportError:
    _HAS_SIMPLEAUDIO = False

# Import unified base classes
try:
    from .base import GenerativeSequencer
    _HAS_UNIFIED_API = True
except ImportError:
    _HAS_UNIFIED_API = False


# Define ragas and their intervals
RAGAS_BY_TIME = {
    "early_morning": {
        "time_range": (4, 7),
        "ragas": [
            {"name": "Bhairav", "intervals": [0, 1, 3, 4, 7, 8, 10, 12]},
            {"name": "Ramkali", "intervals": [0, 1, 3, 4, 7, 8, 10, 12]}
        ]
    },
    "late_morning": {
        "time_range": (7, 10),
        "ragas": [
            {"name": "Bilawal", "intervals": [0, 2, 4, 5, 7, 9, 11, 12]},
            {"name": "Jaunpuri", "intervals": [0, 2, 3, 5, 7, 8, 10, 12]}
        ]
    },
    "afternoon": {
        "time_range": (10, 14),
        "ragas": [
            {"name": "Sarang", "intervals": [0, 2, 4, 7, 9, 12]},
            {"name": "Brindavani Sarang", "intervals": [0, 2, 4, 7, 9, 12]}
        ]
    },
    "evening": {
        "time_range": (17, 22),
        "ragas": [
            {"name": "Yaman", "intervals": [0, 2, 4, 6, 7, 9, 11, 12]},
            {"name": "Kafi", "intervals": [0, 2, 3, 5, 7, 8, 10, 12]}
        ]
    },
    "night": {
        "time_range": (22, 4),
        "ragas": [
            {"name": "Malkauns", "intervals": [0, 3, 5, 8, 10, 12]},
            {"name": "Bageshree", "intervals": [0, 3, 5, 7, 8, 10, 12]}
        ]
    }
}


# =============================================================================
# Unified API (new)
# =============================================================================

if _HAS_UNIFIED_API:
    class RagaSequencer(GenerativeSequencer):
        """
        Raga-based generative sequencer with fractal rhythms.

        Generates music using Indian raga scales with time-appropriate
        raga selection and fractal rhythm patterns.

        Parameters
        ----------
        sample_rate : int
            Audio sample rate (default: 44100)
        bpm : float
            Tempo in BPM (default: 120)
        root_frequency : float
            Root frequency for the scale (default: 220.0, A3)

        Examples
        --------
        >>> seq = RagaSequencer(bpm=90, root_frequency=220.0)
        >>> raga = seq.choose_raga()
        >>> audio = seq.generate_raga_phrase(raga, duration=30.0)
        """

        RAGAS_BY_TIME = RAGAS_BY_TIME

        def __init__(self, sample_rate: int = 44100, bpm: float = 120,
                     root_frequency: float = 220.0):
            super().__init__(sample_rate, bpm)
            self.root_frequency = root_frequency
            self._core_pattern = None

        def choose_raga(self, hour: int = None) -> dict:
            """
            Choose an appropriate raga for the current time.

            Parameters
            ----------
            hour : int, optional
                Hour of day (0-23). Defaults to current hour.

            Returns
            -------
            dict
                Raga dictionary with 'name' and 'intervals'
            """
            if hour is None:
                hour = datetime.now().hour

            for period, details in self.RAGAS_BY_TIME.items():
                start, end = details["time_range"]
                if start < end:
                    if start <= hour < end:
                        return random.choice(details["ragas"])
                else:
                    if hour >= start or hour < end:
                        return random.choice(details["ragas"])
            return None

        def generate_fractal_rhythm(self, core_pattern: list = None,
                                    depth: int = 3, randomness: float = 0.2) -> list:
            """
            Generate a fractal rhythm pattern.

            Parameters
            ----------
            core_pattern : list, optional
                Starting pattern (defaults to random)
            depth : int
                Recursion depth (default: 3)
            randomness : float
                Probability of mutation (default: 0.2)

            Returns
            -------
            list
                Rhythm pattern as list of 0s and 1s
            """
            if core_pattern is None:
                core_pattern = [random.choice([0, 1]) for _ in range(6)]

            pattern = core_pattern[:]
            for _ in range(depth):
                next_pattern = []
                for step in pattern:
                    if random.random() < randomness:
                        step = 1 if step == 0 else 0
                    next_pattern.append(step)
                pattern = pattern + next_pattern

            return pattern

        def select_next_note(self, current_note: int, intervals: list) -> int:
            """
            Stochastically select next note based on proximity.

            Parameters
            ----------
            current_note : int
                Current note interval
            intervals : list
                Available intervals in the raga

            Returns
            -------
            int
                Next note interval
            """
            distances = np.abs(np.array(intervals) - current_note)
            probabilities = np.exp(-distances)
            probabilities /= probabilities.sum()
            return np.random.choice(intervals, p=probabilities)

        def generate(self, duration: float) -> np.ndarray:
            """
            Generate raga-based music for specified duration.

            Parameters
            ----------
            duration : float
                Duration in seconds

            Returns
            -------
            np.ndarray
                Generated audio
            """
            raga = self.choose_raga()
            if raga is None:
                raga = {"name": "Yaman", "intervals": [0, 2, 4, 6, 7, 9, 11, 12]}
            return self.generate_raga_phrase(raga, duration)

        def generate_raga_phrase(self, raga: dict, duration: float) -> np.ndarray:
            """
            Generate a musical phrase using the specified raga.

            Parameters
            ----------
            raga : dict
                Raga with 'name' and 'intervals'
            duration : float
                Duration in seconds

            Returns
            -------
            np.ndarray
                Generated audio
            """
            total_samples = int(duration * self.sample_rate)
            output = np.zeros(total_samples, dtype=np.float32)

            rhythm = self.generate_fractal_rhythm()
            current_note = random.choice(raga["intervals"])
            current_pos = 0
            note_idx = 0

            while current_pos < total_samples:
                step = rhythm[note_idx % len(rhythm)]

                if step == 0:
                    # Rest
                    rest_samples = int(self.beat_duration * self.sample_rate)
                    current_pos += rest_samples
                else:
                    # Play note
                    next_note = self.select_next_note(current_note, raga["intervals"])
                    frequency = self.root_frequency * (2 ** (next_note / 12))

                    tone = self.generate_tone(frequency, self.beat_duration)
                    tone_len = len(tone)

                    end_pos = min(current_pos + tone_len, total_samples)
                    output[current_pos:end_pos] += tone[:end_pos - current_pos]

                    current_note = next_note
                    current_pos += tone_len

                note_idx += 1

            return self._normalize(output)

        def generate_continuous(self, raga: dict = None, duration_seconds: float = 60.0):
            """
            Generate continuous raga music with periodic pattern changes.

            Parameters
            ----------
            raga : dict, optional
                Raga to use (defaults to time-appropriate)
            duration_seconds : float
                Total duration in seconds

            Yields
            ------
            np.ndarray
                Audio chunks
            """
            if raga is None:
                raga = self.choose_raga()

            current_note = random.choice(raga["intervals"])
            self._core_pattern = self.generate_fractal_rhythm()
            pattern_change_interval = 60  # seconds

            start_time = time.time()
            last_pattern_change = start_time
            note_idx = 0

            while (time.time() - start_time) < duration_seconds:
                # Check for pattern change
                if (time.time() - last_pattern_change) > pattern_change_interval:
                    self._core_pattern = self.generate_fractal_rhythm()
                    last_pattern_change = time.time()

                step = self._core_pattern[note_idx % len(self._core_pattern)]

                if step == 1:
                    next_note = self.select_next_note(current_note, raga["intervals"])
                    frequency = self.root_frequency * (2 ** (next_note / 12))
                    tone = self.generate_tone(frequency, self.beat_duration)
                    current_note = next_note
                    yield tone
                else:
                    # Rest - yield silence
                    rest_samples = int(self.beat_duration * self.sample_rate)
                    yield np.zeros(rest_samples, dtype=np.float32)

                note_idx += 1


# =============================================================================
# Legacy API (backwards compatible)
# =============================================================================

# Alias for backwards compatibility
ragas_by_time = RAGAS_BY_TIME


def choose_raga():
    """Choose a raga based on the current time."""
    current_hour = datetime.now().hour
    for period, details in ragas_by_time.items():
        start, end = details["time_range"]
        if start < end:
            if start <= current_hour < end:
                return np.random.choice(details["ragas"])
        else:
            if current_hour >= start or current_hour < end:
                return np.random.choice(details["ragas"])
    return None


def generate_fractal_rhythm(core_pattern, depth=3, randomness=0.2):
    """Generate fractal rhythm with randomness."""
    pattern = core_pattern
    for _ in range(depth):
        next_pattern = []
        for step in pattern:
            if random.random() < randomness:
                step = 1 if step == 0 else 0
            next_pattern.append(step)
        pattern = pattern + next_pattern
    return pattern


def generate_random_core_pattern(length=6):
    """Generate a new random core pattern."""
    return [random.choice([0, 1]) for _ in range(length)]


def select_next_note(current_note, intervals):
    """Stochastic note selection."""
    distances = np.abs(np.array(intervals) - current_note)
    probabilities = np.exp(-distances)
    probabilities /= probabilities.sum()
    next_note = np.random.choice(intervals, p=probabilities)
    return next_note


def generate_sine_wave_with_envelope(frequency, duration, sample_rate=44100, attack=0.01, decay=0.1):
    """Generate sine wave with Attack/Decay envelope."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = np.sin(2 * np.pi * frequency * t)

    envelope = np.ones_like(waveform)
    attack_samples = int(sample_rate * attack)
    decay_samples = int(sample_rate * decay)
    sustain_samples = len(waveform) - attack_samples - decay_samples

    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)

    waveform *= envelope
    return waveform


def play_music_with_fractal_rhythm(raga, root_frequency=440.0, tempo=120):
    """Play music using fractal rhythm and raga notes."""
    if not _HAS_SIMPLEAUDIO:
        print("simpleaudio not available - cannot play audio")
        return

    print(f"Playing Raga: {raga['name']} with intervals {raga['intervals']} at tempo {tempo} BPM")
    sample_rate = 44100
    core_pattern = generate_random_core_pattern()
    fractal_rhythm = generate_fractal_rhythm(core_pattern, depth=3)

    current_note = np.random.choice(raga["intervals"])
    next_core_change = datetime.now() + timedelta(minutes=5)
    beat_duration = 60 / tempo

    for step in fractal_rhythm:
        if datetime.now() >= next_core_change:
            core_pattern = generate_random_core_pattern()
            fractal_rhythm = generate_fractal_rhythm(core_pattern, depth=3)
            next_core_change = datetime.now() + timedelta(minutes=5)

        if step == 0:
            time.sleep(beat_duration)
            continue

        next_note = select_next_note(current_note, raga["intervals"])
        frequency = root_frequency * (2 ** (next_note / 12))

        waveform = generate_sine_wave_with_envelope(frequency, beat_duration, sample_rate)
        waveform = (waveform * 32767).astype(np.int16)

        play_obj = sa.play_buffer(waveform, 1, 2, sample_rate)
        play_obj.wait_done()

        current_note = next_note


def run_music_box():
    """Run the dynamic music box."""
    while True:
        raga = choose_raga()
        if raga:
            play_music_with_fractal_rhythm(raga, root_frequency=220.0, tempo=300)
        else:
            print("No raga found for the current time.")
            time.sleep(60)


if __name__ == "__main__":
    run_music_box()
