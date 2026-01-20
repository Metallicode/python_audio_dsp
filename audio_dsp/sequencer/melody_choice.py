"""
Interactive melody development and counterpoint generation.

Provides tools for composing melodies through binary choice interaction
and developing them using techniques like mirroring, inversion, and
repetition. Includes automatic counterpoint voice generation.

This module provides both the legacy functional API and the new unified
class-based API via MelodySequencer.
"""

import numpy as np
import random
import os

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
BAR_DURATION = 4.0

# Major scale
BASE_FREQ = 261.63  # C4
SCALE_STEPS = [0, 2, 4, 5, 7, 9, 11]  # C major scale
DURATIONS = [0.25, 0.5, 0.75, 1.0]


# =============================================================================
# Unified API (new)
# =============================================================================

if _HAS_UNIFIED_API:
    class MelodySequencer(GenerativeSequencer):
        """
        Melodic composition and development sequencer.

        Creates melodies using development techniques like mirroring,
        inversion, and repetition. Can generate counterpoint voices.

        Parameters
        ----------
        sample_rate : int
            Audio sample rate (default: 44100)
        bpm : float
            Tempo in BPM (default: 120)
        base_freq : float
            Base frequency (default: 261.63, C4)
        scale_steps : list
            Scale intervals in semitones (default: major scale)

        Examples
        --------
        >>> seq = MelodySequencer(bpm=90)
        >>> melody = seq.generate_random_melody(length=8)
        >>> developed = seq.develop_melody(melody, target_bars=16)
        >>> audio = seq.render_melody(developed)
        """

        def __init__(self, sample_rate: int = 44100, bpm: float = 120,
                     base_freq: float = 261.63, scale_steps: list = None):
            super().__init__(sample_rate, bpm)
            self.base_freq = base_freq
            self.scale_steps = scale_steps or SCALE_STEPS
            self.bar_duration = 60 / bpm * 4

        def generate_random_melody(self, length: int = 8,
                                   durations: list = None) -> list:
            """
            Generate a random melody.

            Parameters
            ----------
            length : int
                Number of notes (default: 8)
            durations : list
                Available durations in beats (default: [0.25, 0.5, 0.75, 1.0])

            Returns
            -------
            list
                List of (frequency, duration) tuples
            """
            durations = durations or DURATIONS
            melody = []
            current_freq = self.base_freq

            for _ in range(length):
                step = random.choice(self.scale_steps)
                freq = current_freq * (2 ** (step / 12))
                dur = random.choice(durations)
                melody.append((freq, dur))
                current_freq = freq

            return melody

        def horizontal_mirror(self, melody: list) -> list:
            """Reverse the melody (retrograde)."""
            return melody[::-1]

        def vertical_flip(self, melody: list) -> list:
            """Invert the melody around the base frequency."""
            return [(self.base_freq * (self.base_freq / freq), dur)
                    for freq, dur in melody]

        def repeat_segment(self, melody: list) -> list:
            """Repeat a random segment of the melody."""
            if len(melody) < 2:
                return melody
            start = random.randint(0, len(melody) - 2)
            segment = melody[start:start + 2]
            return melody[:start] + segment + segment + melody[start + 2:]

        def develop_melody(self, base_melody: list, target_bars: int = 16) -> list:
            """
            Develop a melody to target length using compositional techniques.

            Parameters
            ----------
            base_melody : list
                Starting melody as list of (freq, duration) tuples
            target_bars : int
                Target length in bars (default: 16)

            Returns
            -------
            list
                Developed melody
            """
            techniques = [self.horizontal_mirror, self.vertical_flip, self.repeat_segment]
            melody = base_melody[:]
            total_duration = sum(dur for _, dur in melody)
            target_duration = target_bars * self.bar_duration

            while total_duration < target_duration:
                technique = random.choice(techniques)
                melody = technique(melody)
                total_duration = sum(dur for _, dur in melody)

            # Trim if too long
            if total_duration > target_duration:
                excess = total_duration - target_duration
                for i in range(len(melody) - 1, -1, -1):
                    if excess > 0:
                        dur = melody[i][1]
                        if dur <= excess:
                            excess -= dur
                            melody.pop(i)
                        else:
                            melody[i] = (melody[i][0], dur - excess)
                            excess = 0
            elif total_duration < target_duration:
                melody.append((self.base_freq, target_duration - total_duration))

            return melody

        def generate_counterpoint(self, melody: list, voice_shift: int = 1) -> list:
            """
            Generate a contrapuntal voice from an existing melody.

            Parameters
            ----------
            melody : list
                Source melody
            voice_shift : int
                Direction and amount of shift (-1, 1, etc.)

            Returns
            -------
            list
                Counterpoint melody
            """
            counterpoint = []
            for freq, dur in melody:
                # Map to nearest scale step
                semitone = round(12 * np.log2(freq / self.base_freq))
                step_shift = (semitone + random.choice([-2, -1, 1, 2, 3, 4]) * voice_shift) % 12
                nearest_step = min(self.scale_steps, key=lambda x: abs(x - step_shift))
                new_freq = self.base_freq * (2 ** (nearest_step / 12))
                counterpoint.append((new_freq, dur))
            return counterpoint

        def render_melody(self, melody: list) -> np.ndarray:
            """
            Render a melody to audio.

            Parameters
            ----------
            melody : list
                List of (frequency, duration) tuples

            Returns
            -------
            np.ndarray
                Rendered audio
            """
            total_duration = sum(dur for _, dur in melody)
            total_samples = int(total_duration * self.sample_rate)
            output = np.zeros(total_samples, dtype=np.float32)

            current_pos = 0
            for freq, dur in melody:
                tone = self.generate_tone(freq, dur)
                tone_len = len(tone)
                end_pos = min(current_pos + tone_len, total_samples)
                output[current_pos:end_pos] += tone[:end_pos - current_pos]
                current_pos += tone_len

            return self._normalize(output)

        def render_voices(self, voices: list) -> np.ndarray:
            """
            Render multiple voices and mix them.

            Parameters
            ----------
            voices : list
                List of melodies (each a list of (freq, dur) tuples)

            Returns
            -------
            np.ndarray
                Mixed audio
            """
            rendered = [self.render_melody(voice) for voice in voices]
            max_len = max(len(v) for v in rendered)
            output = np.zeros(max_len, dtype=np.float32)

            for voice in rendered:
                output[:len(voice)] += voice / len(voices)

            return self._normalize(output)

        def generate(self, duration: float) -> np.ndarray:
            """Generate melody composition for specified duration."""
            bars = int(duration / self.bar_duration) or 1
            melody = self.generate_random_melody(8)
            developed = self.develop_melody(melody, target_bars=bars)
            return self.render_melody(developed)


# =============================================================================
# Legacy API (backwards compatible)
# =============================================================================

def generate_tone(freq, duration):
    """Generate a tone with envelope."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.sin(freq * t * 2 * np.pi) * 0.5
    fade_samples = int(SAMPLE_RATE * duration * 0.1)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    if fade_samples < len(wave):
        wave[:fade_samples] *= fade_in
        wave[-fade_samples:] *= fade_out
    return wave.astype(np.float32)


def play_audio(wave):
    """Play audio via temp file."""
    temp_file = "temp.wav"
    if wavfile:
        wavfile.write(temp_file, SAMPLE_RATE, (wave * 32767).astype(np.int16))
    if os.name == 'nt':
        import winsound
        winsound.PlaySound(temp_file, winsound.SND_FILENAME)
    else:
        os.system(f"afplay {temp_file}" if os.uname().sysname == 'Darwin' else f"aplay {temp_file}")
    os.remove(temp_file)


def generate_option_tone(current_freq):
    """Generate a random option tone."""
    step = random.choice(SCALE_STEPS)
    freq = current_freq * (2 ** (step / 12))
    duration = random.choice(DURATIONS)
    return freq, duration


def horizontal_mirror(melody):
    """Reverse melody."""
    return melody[::-1]


def vertical_flip(melody, pivot_freq=BASE_FREQ):
    """Invert melody."""
    return [(pivot_freq * (pivot_freq / freq), dur) for freq, dur in melody]


def repeat_segment(melody):
    """Repeat a segment."""
    if len(melody) < 2:
        return melody
    start = random.randint(0, len(melody) - 2)
    segment = melody[start:start + 2]
    return melody[:start] + segment + segment + melody[start + 2:]


DEVELOPMENTS = [horizontal_mirror, vertical_flip, repeat_segment]


def develop_melody(base_melody, target_bars=16):
    """Develop melody to target length."""
    melody = base_melody[:]
    total_duration = sum(dur for _, dur in melody)
    target_duration = target_bars * BAR_DURATION

    while total_duration < target_duration:
        technique = random.choice(DEVELOPMENTS)
        melody = technique(melody)
        total_duration = sum(dur for _, dur in melody)

    if total_duration > target_duration:
        excess = total_duration - target_duration
        for i in range(len(melody) - 1, -1, -1):
            if excess > 0:
                dur = melody[i][1]
                if dur <= excess:
                    excess -= dur
                    melody.pop(i)
                else:
                    melody[i] = (melody[i][0], dur - excess)
                    excess = 0
    elif total_duration < target_duration:
        melody.append((BASE_FREQ, target_duration - total_duration))

    return melody


def generate_counterpoint(melody, voice_num):
    """Generate a contrapuntal voice."""
    counterpoint = []
    for freq, dur in melody:
        semitone = round(12 * np.log2(freq / BASE_FREQ))
        step_shift = (semitone + random.choice([-2, -1, 1, 2, 3, 4]) * voice_num) % 12
        nearest_step = min(SCALE_STEPS, key=lambda x: abs(x - step_shift))
        new_freq = BASE_FREQ * (2 ** (nearest_step / 12))
        counterpoint.append((new_freq, dur))
    return counterpoint


def main():
    """Interactive melody composition."""
    melody = [(BASE_FREQ, 1.0)]
    print("Starting melody with C4 (261.63 Hz, 1.0s)")
    play_audio(generate_tone(BASE_FREQ, 1.0))

    while True:
        last_freq = melody[-1][0]
        freq1, dur1 = generate_option_tone(last_freq)
        freq2, dur2 = generate_option_tone(last_freq)

        print(f"\nOption 1: Frequency {freq1:.2f} Hz, Duration {dur1}s")
        play_audio(generate_tone(freq1, dur1))
        print(f"Option 2: Frequency {freq2:.2f} Hz, Duration {dur2}s")
        play_audio(generate_tone(freq2, dur2))

        choice = input("Choose (1 or 2) or 'done': ").strip().lower()
        if choice == 'done':
            break
        elif choice not in ['1', '2']:
            print("Invalid choice.")
            continue

        chosen_freq, chosen_dur = (freq1, dur1) if choice == '1' else (freq2, dur2)
        melody.append((chosen_freq, chosen_dur))
        print(f"Added: {chosen_freq:.2f} Hz, {chosen_dur}s")
        play_audio(np.concatenate([generate_tone(f, d) for f, d in melody]))

    # Develop melody
    bpm = float(input("Enter tempo in BPM (e.g., 120): "))
    global BAR_DURATION
    BAR_DURATION = 60 / bpm * 4
    developed_melody = develop_melody(melody)
    print(f"Developed melody: {[(f'{f:.2f}', d) for f, d in developed_melody]}")

    # Generate voices
    voice1 = developed_melody
    voice2 = generate_counterpoint(voice1, 1)
    voice3 = generate_counterpoint(voice1, -1)

    print(f"Voice 2: {[(f'{f:.2f}', d) for f, d in voice2[:5]]}...")
    print(f"Voice 3: {[(f'{f:.2f}', d) for f, d in voice3[:5]]}...")

    # Generate audio
    voice1_wave = np.concatenate([generate_tone(f, d) for f, d in voice1])
    voice2_wave = np.concatenate([generate_tone(f, d) for f, d in voice2])
    voice3_wave = np.concatenate([generate_tone(f, d) for f, d in voice3])

    # Mix
    max_len = max(len(voice1_wave), len(voice2_wave), len(voice3_wave))
    mixed_wave = np.zeros(max_len, dtype=np.float32)
    for wave in [voice1_wave, voice2_wave, voice3_wave]:
        mixed_wave[:len(wave)] += wave / 3

    # Save
    print("Saving to 'three_voice_composition.wav'")
    if wavfile:
        wavfile.write("three_voice_composition.wav", SAMPLE_RATE, (mixed_wave * 32767).astype(np.int16))
    print("Playing final composition...")
    play_audio(mixed_wave)
    print("Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
