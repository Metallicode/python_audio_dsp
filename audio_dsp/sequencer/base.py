"""
Base sequencer classes providing a unified interface for all sequencer types.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Generator, Optional, Union
import soundfile as sf


class BaseSequencer(ABC):
    """
    Abstract base class for all sequencers.

    Provides common functionality for timing, audio generation, and export.
    Subclasses must implement the generate() method.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz (default: 44100)
    bpm : float
        Tempo in beats per minute (default: 120)
    """

    def __init__(self, sample_rate: int = 44100, bpm: float = 120):
        self.sample_rate = sample_rate
        self.bpm = bpm

    @property
    def beat_duration(self) -> float:
        """Duration of one beat in seconds."""
        return 60.0 / self.bpm

    @property
    def step_duration(self) -> float:
        """Duration of one step (1/16th note) in seconds."""
        return self.beat_duration / 4

    def beats_to_samples(self, beats: float) -> int:
        """Convert beats to samples."""
        return int(beats * self.beat_duration * self.sample_rate)

    def seconds_to_samples(self, seconds: float) -> int:
        """Convert seconds to samples."""
        return int(seconds * self.sample_rate)

    def samples_to_seconds(self, samples: int) -> float:
        """Convert samples to seconds."""
        return samples / self.sample_rate

    @abstractmethod
    def generate(self, duration: float) -> np.ndarray:
        """
        Generate audio for the specified duration.

        Parameters
        ----------
        duration : float
            Duration in seconds

        Returns
        -------
        np.ndarray
            Generated audio as a numpy array (float32, normalized to [-1, 1])
        """
        pass

    def stream(self, chunk_size: int = 4096) -> Generator[np.ndarray, None, None]:
        """
        Yield audio chunks for real-time streaming.

        Default implementation generates chunks indefinitely.
        Override in subclasses for custom streaming behavior.

        Parameters
        ----------
        chunk_size : int
            Number of samples per chunk (default: 4096)

        Yields
        ------
        np.ndarray
            Audio chunks
        """
        while True:
            chunk_duration = chunk_size / self.sample_rate
            yield self.generate(chunk_duration)

    def export(self, filepath: str, duration: float, subtype: str = 'PCM_16'):
        """
        Export generated audio to a WAV file.

        Parameters
        ----------
        filepath : str
            Output file path
        duration : float
            Duration in seconds
        subtype : str
            Audio format subtype (default: 'PCM_16')
        """
        audio = self.generate(duration)
        audio = self._normalize(audio)
        sf.write(filepath, audio, self.sample_rate, subtype=subtype)
        print(f"Exported to {filepath} (duration: {duration:.2f}s)")

    @staticmethod
    def _normalize(audio: np.ndarray, headroom_db: float = -1.0) -> np.ndarray:
        """
        Normalize audio to prevent clipping.

        Parameters
        ----------
        audio : np.ndarray
            Input audio
        headroom_db : float
            Headroom in dB below 0 (default: -1.0)

        Returns
        -------
        np.ndarray
            Normalized audio
        """
        max_amp = np.max(np.abs(audio))
        if max_amp > 0:
            target = 10 ** (headroom_db / 20)
            audio = audio * (target / max_amp)
        return audio.astype(np.float32)


class PatternBasedSequencer(BaseSequencer):
    """
    Base class for pattern-based sequencers (e.g., drum sequencers).

    Extends BaseSequencer with pattern parsing and sample triggering.
    """

    def __init__(self, sample_rate: int = 44100, bpm: float = 120):
        super().__init__(sample_rate, bpm)
        self._sample_manager = None

    @property
    def sample_manager(self):
        """Lazy-loaded sample manager."""
        if self._sample_manager is None:
            from .sample_manager import SampleManager
            self._sample_manager = SampleManager(sample_rate=self.sample_rate)
        return self._sample_manager

    def add_sample(self, name: str, audio_or_path: Union[np.ndarray, str],
                   sr: Optional[int] = None):
        """
        Add a sample to the sequencer.

        Parameters
        ----------
        name : str
            Sample name/identifier
        audio_or_path : np.ndarray or str
            Either a numpy array containing audio data, or a path to a WAV file
        sr : int, optional
            Sample rate of the audio (required if audio_or_path is np.ndarray)
        """
        if isinstance(audio_or_path, np.ndarray):
            self.sample_manager.add(name, audio_or_path, sr or self.sample_rate)
        else:
            self.sample_manager.load(audio_or_path, name)

    def parse_pattern(self, pattern: str) -> list:
        """
        Parse a pattern string into a list of trigger positions.

        Parameters
        ----------
        pattern : str
            Pattern string (e.g., "1010101010101010")

        Returns
        -------
        list
            List of boolean trigger values
        """
        return [c == '1' for c in pattern.replace(' ', '')]


class GenerativeSequencer(BaseSequencer):
    """
    Base class for generative sequencers that synthesize audio algorithmically.

    These sequencers typically generate audio using synthesis rather than samples.
    """

    def __init__(self, sample_rate: int = 44100, bpm: float = 120):
        super().__init__(sample_rate, bpm)

    def generate_tone(self, frequency: float, duration: float,
                      attack: float = 0.01, decay: float = 0.1) -> np.ndarray:
        """
        Generate a sine tone with an ADSR-style envelope.

        Parameters
        ----------
        frequency : float
            Frequency in Hz
        duration : float
            Duration in seconds
        attack : float
            Attack time in seconds (default: 0.01)
        decay : float
            Decay/release time in seconds (default: 0.1)

        Returns
        -------
        np.ndarray
            Generated tone
        """
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples, endpoint=False)
        waveform = np.sin(2 * np.pi * frequency * t)

        # Create envelope
        envelope = np.ones(samples, dtype=np.float32)
        attack_samples = int(self.sample_rate * min(attack, duration * 0.4))
        decay_samples = int(self.sample_rate * min(decay, duration * 0.4))

        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        if decay_samples > 0 and decay_samples < samples:
            envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)

        return (waveform * envelope).astype(np.float32)

    def freq_to_midi(self, frequency: float) -> float:
        """Convert frequency to MIDI note number."""
        return 69 + 12 * np.log2(frequency / 440.0)

    def midi_to_freq(self, midi_note: float) -> float:
        """Convert MIDI note number to frequency."""
        return 440.0 * (2 ** ((midi_note - 69) / 12))
