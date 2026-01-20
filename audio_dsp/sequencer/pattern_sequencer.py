"""
Unified pattern-based sequencer for sample playback.

Provides a consistent API for triggering samples based on text patterns.
Supports both file-based samples and numpy arrays from synths/effects.
"""

import numpy as np
import re
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .base import PatternBasedSequencer
from .sample_manager import SampleManager


class PatternSequencer(PatternBasedSequencer):
    """
    A pattern-based sequencer that triggers samples at specified positions.

    Supports multiple tracks with independent patterns and timing modifiers.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate (default: 44100)
    bpm : float
        Tempo in beats per minute (default: 120)

    Examples
    --------
    >>> from audio_dsp.synth import SubtractiveSynth
    >>> from audio_dsp.sequencer import PatternSequencer
    >>>
    >>> # Create sequencer
    >>> seq = PatternSequencer(bpm=120)
    >>>
    >>> # Add samples from synth
    >>> synth = SubtractiveSynth()
    >>> seq.add_sample("kick", synth.synthesize(60, 0.1))
    >>> seq.add_sample("snare", synth.synthesize(200, 0.1))
    >>>
    >>> # Or load from files
    >>> seq.add_sample("hihat", "samples/hihat.wav")
    >>>
    >>> # Generate with patterns
    >>> patterns = {
    ...     "kick": "1000100010001000",
    ...     "snare": "0000100000001000",
    ...     "hihat": "1010101010101010"
    ... }
    >>> audio = seq.generate_from_patterns(patterns, duration=4.0)
    """

    def __init__(self, sample_rate: int = 44100, bpm: float = 120):
        super().__init__(sample_rate, bpm)
        self.tracks: Dict[str, dict] = {}

    def add_track(self, name: str, pattern: str, sample_name: Optional[str] = None,
                  volume: float = 1.0, modifier: Optional[str] = None):
        """
        Add a track to the sequencer.

        Parameters
        ----------
        name : str
            Track name
        pattern : str
            Trigger pattern (e.g., "1000100010001000")
        sample_name : str, optional
            Name of sample to use (defaults to track name)
        volume : float
            Track volume (0.0 to 1.0, default: 1.0)
        modifier : str, optional
            Timing modifier (e.g., "*2" for double speed, "/2" for half speed)
        """
        self.tracks[name] = {
            "pattern": pattern.replace(" ", ""),
            "sample": sample_name or name,
            "volume": volume,
            "modifier": self._parse_modifier(modifier)
        }

    def remove_track(self, name: str):
        """Remove a track by name."""
        if name in self.tracks:
            del self.tracks[name]

    def clear_tracks(self):
        """Remove all tracks."""
        self.tracks.clear()

    def generate(self, duration: float) -> np.ndarray:
        """
        Generate audio for the specified duration using configured tracks.

        Parameters
        ----------
        duration : float
            Duration in seconds

        Returns
        -------
        np.ndarray
            Generated audio
        """
        if not self.tracks:
            return np.zeros(int(duration * self.sample_rate), dtype=np.float32)

        return self._generate_from_tracks(duration)

    def generate_from_patterns(self, patterns: Dict[str, str],
                               duration: float,
                               volumes: Optional[Dict[str, float]] = None,
                               modifiers: Optional[Dict[str, str]] = None) -> np.ndarray:
        """
        Generate audio from a dictionary of patterns.

        Parameters
        ----------
        patterns : dict
            Mapping of sample names to pattern strings
        duration : float
            Duration in seconds
        volumes : dict, optional
            Mapping of sample names to volumes (0.0 to 1.0)
        modifiers : dict, optional
            Mapping of sample names to timing modifiers

        Returns
        -------
        np.ndarray
            Generated audio
        """
        volumes = volumes or {}
        modifiers = modifiers or {}

        # Configure tracks
        self.clear_tracks()
        for name, pattern in patterns.items():
            self.add_track(
                name=name,
                pattern=pattern,
                sample_name=name,
                volume=volumes.get(name, 1.0),
                modifier=modifiers.get(name)
            )

        return self.generate(duration)

    def generate_from_string(self, pattern_string: str, duration: float) -> np.ndarray:
        """
        Generate audio from a single pattern string.

        The pattern string maps characters to sample names:
        - '1' or trigger character triggers the sample
        - '0' or '.' is a rest

        Parameters
        ----------
        pattern_string : str
            Pattern string (e.g., "1010101010101010")
        duration : float
            Duration in seconds

        Returns
        -------
        np.ndarray
            Generated audio
        """
        total_samples = int(duration * self.sample_rate)
        output = np.zeros(total_samples, dtype=np.float32)

        # Use first available sample
        if not self.sample_manager.list_samples():
            print("Warning: No samples loaded")
            return output

        sample_name = self.sample_manager.list_samples()[0]
        return self._render_pattern_to_buffer(
            output, pattern_string, sample_name, 1.0, ("", 1), total_samples
        )

    def _generate_from_tracks(self, duration: float) -> np.ndarray:
        """Generate audio from configured tracks."""
        total_samples = int(duration * self.sample_rate)
        output = np.zeros(total_samples, dtype=np.float32)

        # Calculate max duration based on patterns
        for track_name, track_info in self.tracks.items():
            pattern = track_info["pattern"]
            sample_name = track_info["sample"]
            volume = track_info["volume"]
            modifier = track_info["modifier"]

            if sample_name not in self.sample_manager:
                print(f"Warning: Sample '{sample_name}' not found for track '{track_name}'")
                continue

            output = self._render_pattern_to_buffer(
                output, pattern, sample_name, volume, modifier, total_samples
            )

        return self._normalize(output)

    def _render_pattern_to_buffer(self, output: np.ndarray, pattern: str,
                                  sample_name: str, volume: float,
                                  modifier: Tuple[str, int],
                                  total_samples: int) -> np.ndarray:
        """Render a pattern to the output buffer."""
        op, factor = modifier
        step_samples = self._get_step_samples(op, factor)

        # Extend pattern to cover duration
        pattern_samples = len(pattern) * step_samples
        if pattern_samples > 0:
            repeats = (total_samples // pattern_samples) + 1
            extended_pattern = (pattern * repeats)

        sample = self.sample_manager.get(sample_name) * volume
        sample_len = len(sample)

        for step, trigger in enumerate(extended_pattern):
            if trigger == '1':
                start = step * step_samples
                if start >= total_samples:
                    break
                end = min(start + sample_len, total_samples)
                sample_segment = sample[:end - start]
                output[start:end] += sample_segment

        return output

    def _get_step_samples(self, op: str, factor: int) -> int:
        """Get step duration in samples based on modifier."""
        base_step_duration = self.step_duration

        if op == "*":
            step_duration = base_step_duration / factor
        elif op == "/":
            step_duration = base_step_duration * factor
        else:
            step_duration = base_step_duration

        return int(step_duration * self.sample_rate)

    @staticmethod
    def _parse_modifier(modifier: Optional[str]) -> Tuple[str, int]:
        """Parse a timing modifier string."""
        if not modifier:
            return ("", 1)

        match = re.match(r"([*/])(\d+)", modifier)
        if match:
            return (match.group(1), int(match.group(2)))
        return ("", 1)

    @classmethod
    def from_pattern_file(cls, pattern_file: str, samples_dir: str,
                          bpm: float = 120) -> "PatternSequencer":
        """
        Create a PatternSequencer from a pattern file.

        Parameters
        ----------
        pattern_file : str
            Path to pattern file
        samples_dir : str
            Path to samples directory
        bpm : float
            Tempo in BPM

        Returns
        -------
        PatternSequencer
            Configured sequencer

        Pattern file format:
        ```
        1010101010101010
        0000100000001000 *2
        1000100010001000 /2
        [1.0, 0.8, 0.6]
        ```
        """
        seq = cls(bpm=bpm)

        # Parse pattern file
        patterns, volumes, modifiers = cls._parse_pattern_file(pattern_file)

        # Load samples (expecting 1_sample.wav, 2_sample.wav, etc.)
        seq.sample_manager.load_directory(samples_dir)

        # Configure tracks
        for i, (pattern, vol, mod) in enumerate(zip(patterns, volumes, modifiers), 1):
            # Find sample matching track number
            sample_name = None
            for name in seq.sample_manager.list_samples():
                if name.startswith(f"{i}_") or name == str(i):
                    sample_name = name
                    break

            if sample_name:
                seq.add_track(
                    name=f"track_{i}",
                    pattern=pattern,
                    sample_name=sample_name,
                    volume=vol,
                    modifier=f"{mod[0]}{mod[1]}" if mod[0] else None
                )

        return seq

    @staticmethod
    def _parse_pattern_file(pattern_file: str) -> Tuple[List[str], List[float], List[Tuple[str, int]]]:
        """Parse a pattern file into patterns, volumes, and modifiers."""
        with open(pattern_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines or not lines[-1].startswith("["):
            raise ValueError("Pattern file must end with a volume list like [1.0, 0.8, ...]")

        patterns = lines[:-1]
        volume_line = lines[-1]

        # Parse volumes
        try:
            volumes = ast.literal_eval(volume_line)
            if not isinstance(volumes, list):
                raise ValueError("Volume line must be a list")
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid volume list format: {e}")

        # Parse patterns and modifiers
        parsed_patterns = []
        modifiers = []
        for p in patterns:
            match = re.match(r"^(.*?)([*/]\d+)?$", p.replace(" ", ""))
            if not match:
                raise ValueError(f"Invalid pattern format: {p}")
            pattern, modifier = match.groups()
            if modifier:
                op = modifier[0]
                factor = int(modifier[1:])
                modifiers.append((op, factor))
            else:
                modifiers.append(("", 1))
            parsed_patterns.append(pattern)

        # Extend volumes if needed
        while len(volumes) < len(parsed_patterns):
            volumes.append(1.0)

        return parsed_patterns, volumes, modifiers


class LiquidSequencer(PatternSequencer):
    """
    Pattern sequencer with non-linear ("liquid") timing.

    Adds randomized timing offsets, swing, and dynamics for a more
    organic, human feel.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate (default: 44100)
    bpm : float
        Tempo in BPM (default: 120)
    swing : float
        Swing amount (0.0 to 1.0, default: 0.1)
    jitter : float
        Random timing jitter amount (default: 0.05)
    """

    def __init__(self, sample_rate: int = 44100, bpm: float = 120,
                 swing: float = 0.1, jitter: float = 0.05):
        super().__init__(sample_rate, bpm)
        self.swing = swing
        self.jitter = jitter

    def generate_liquid_timing(self, pattern: str, loop_length: float = 4.0) -> np.ndarray:
        """
        Generate non-linear timing for a pattern.

        Parameters
        ----------
        pattern : str
            Pattern string
        loop_length : float
            Loop length in beats (default: 4.0)

        Returns
        -------
        np.ndarray
            Array of trigger times in seconds
        """
        import random

        num_events = len(pattern)
        loop_duration = loop_length * self.beat_duration
        grid_times = np.linspace(0, loop_duration, num_events, endpoint=False)

        liquid_times = []
        for i, grid_time in enumerate(grid_times):
            # Base offset with damped oscillation
            A = (random.uniform(0.1, 0.6)) * self.beat_duration
            omega = 2 * np.pi * 0.5
            gamma = 0.1
            phi = random.uniform(0, 2 * np.pi)
            offset = A * np.sin(omega * grid_time + phi) * np.exp(-gamma * grid_time)

            # Add jitter
            jitter_offset = random.gauss(0, self.jitter * self.beat_duration)

            # Add swing to off-beats
            swing_offset = self.swing * self.beat_duration if i % 2 == 1 else 0

            liquid_time = grid_time + offset + jitter_offset + swing_offset
            liquid_time = np.clip(liquid_time, 0, loop_duration - 0.001)
            liquid_times.append(liquid_time)

        return np.sort(np.array(liquid_times))

    def generate_with_liquid_timing(self, pattern: str, sample_name: str,
                                    loops: int = 1, loop_length: float = 4.0,
                                    volume: float = 1.0) -> np.ndarray:
        """
        Generate audio with liquid (non-linear) timing.

        Parameters
        ----------
        pattern : str
            Pattern string (e.g., "K.S.H.K.")
        sample_name : str
            Name of sample to trigger
        loops : int
            Number of loop repetitions (default: 1)
        loop_length : float
            Loop length in beats (default: 4.0)
        volume : float
            Volume (default: 1.0)

        Returns
        -------
        np.ndarray
            Generated audio with liquid timing
        """
        import random

        loop_duration = loop_length * self.beat_duration
        samples_per_loop = int(loop_duration * self.sample_rate)
        total_samples = samples_per_loop * loops
        output = np.zeros(total_samples, dtype=np.float32)

        if sample_name not in self.sample_manager:
            print(f"Warning: Sample '{sample_name}' not found")
            return output

        sample = self.sample_manager.get(sample_name)

        for loop in range(loops):
            loop_offset = loop * samples_per_loop
            liquid_times = self.generate_liquid_timing(pattern, loop_length)

            for i, (char, time) in enumerate(zip(pattern, liquid_times)):
                if char not in ['0', '.', ' ']:
                    sample_pos = int(time * self.sample_rate)
                    start = sample_pos + loop_offset
                    end = start + len(sample)

                    if end <= total_samples:
                        # Random velocity
                        velocity = random.uniform(0.5, 1.0) * volume
                        output[start:end] += sample * velocity

        return self._normalize(output)
