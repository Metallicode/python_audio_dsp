"""
Unified sample management for sequencers.

Provides loading, caching, and retrieval of audio samples from various sources:
- File paths (WAV, FLAC, etc.)
- Numpy arrays (from synths, effects, or other processing)
- Default sample library
"""

import numpy as np
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
from glob import glob


def get_default_samples_dir() -> Path:
    """Get the path to the default samples directory."""
    return Path(__file__).parent / "samples"


class SampleManager:
    """
    Manages audio samples for sequencers.

    Supports loading from files or adding numpy arrays directly.
    All samples are automatically resampled to the manager's sample rate.

    Parameters
    ----------
    samples_dir : str or Path, optional
        Default directory for loading samples
    sample_rate : int
        Target sample rate for all samples (default: 44100)

    Examples
    --------
    >>> manager = SampleManager(sample_rate=44100)
    >>> # Load from file
    >>> manager.load("drums/kick.wav", "kick")
    >>> # Add numpy array from synth
    >>> manager.add("snare", synth_output, sr=44100)
    >>> # Get sample for playback
    >>> kick_audio = manager.get("kick")
    """

    def __init__(self, samples_dir: Optional[Union[str, Path]] = None,
                 sample_rate: int = 44100):
        self.samples: Dict[str, Tuple[int, np.ndarray]] = {}
        self.default_dir = Path(samples_dir) if samples_dir else get_default_samples_dir()
        self.sample_rate = sample_rate

    def add(self, name: str, audio: np.ndarray, sr: Optional[int] = None):
        """
        Add a numpy array as a sample.

        Useful for adding output from synths or effects as samples.

        Parameters
        ----------
        name : str
            Sample name/identifier
        audio : np.ndarray
            Audio data (will be converted to mono if stereo)
        sr : int, optional
            Sample rate of the audio (defaults to manager's sample_rate)
        """
        sr = sr or self.sample_rate

        # Ensure float32
        if audio.dtype != np.float32:
            if audio.dtype in (np.int16, np.int32):
                audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
            else:
                audio = audio.astype(np.float32)

        # Convert to mono if stereo
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        elif len(audio.shape) > 1:
            audio = audio.flatten()

        self.samples[name] = (sr, audio)

    def load(self, path: Union[str, Path], name: Optional[str] = None) -> str:
        """
        Load a sample from a file path.

        Parameters
        ----------
        path : str or Path
            Path to the audio file (absolute or relative to default_dir)
        name : str, optional
            Name to assign to the sample (defaults to filename without extension)

        Returns
        -------
        str
            The name assigned to the sample
        """
        path = Path(path)

        # Try absolute path first, then relative to default_dir
        if not path.is_absolute():
            if not path.exists():
                path = self.default_dir / path

        if not path.exists():
            raise FileNotFoundError(f"Sample file not found: {path}")

        # Determine sample name
        if name is None:
            name = path.stem.lower()

        # Load audio using soundfile
        try:
            import soundfile as sf
            audio, sr = sf.read(str(path), dtype='float32')
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}")

        self.add(name, audio, sr)
        return name

    def load_directory(self, path: Optional[Union[str, Path]] = None,
                       pattern: str = "*.wav",
                       prefix: str = "") -> List[str]:
        """
        Load all samples from a directory.

        Parameters
        ----------
        path : str or Path, optional
            Directory path (defaults to default_dir)
        pattern : str
            Glob pattern for files (default: "*.wav")
        prefix : str
            Prefix to add to sample names (default: "")

        Returns
        -------
        list
            List of loaded sample names
        """
        path = Path(path) if path else self.default_dir
        loaded = []

        for file_path in glob(str(path / pattern)):
            file_path = Path(file_path)
            name = prefix + file_path.stem.lower()
            try:
                self.load(file_path, name)
                loaded.append(name)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")

        return loaded

    def get(self, name: str, resample: bool = True) -> np.ndarray:
        """
        Get a sample by name, optionally resampled to manager's sample rate.

        Parameters
        ----------
        name : str
            Sample name
        resample : bool
            Whether to resample to manager's sample_rate (default: True)

        Returns
        -------
        np.ndarray
            Audio data
        """
        if name not in self.samples:
            raise KeyError(f"Sample '{name}' not found. Available: {list(self.samples.keys())}")

        sr, audio = self.samples[name]

        if resample and sr != self.sample_rate:
            audio = self._resample(audio, sr, self.sample_rate)

        return audio

    def get_with_sr(self, name: str) -> Tuple[int, np.ndarray]:
        """
        Get a sample with its original sample rate.

        Parameters
        ----------
        name : str
            Sample name

        Returns
        -------
        tuple
            (sample_rate, audio_data)
        """
        if name not in self.samples:
            raise KeyError(f"Sample '{name}' not found. Available: {list(self.samples.keys())}")

        return self.samples[name]

    def list_samples(self) -> List[str]:
        """Get list of all loaded sample names."""
        return list(self.samples.keys())

    def clear(self):
        """Remove all loaded samples."""
        self.samples.clear()

    def remove(self, name: str):
        """Remove a sample by name."""
        if name in self.samples:
            del self.samples[name]

    def __contains__(self, name: str) -> bool:
        """Check if a sample is loaded."""
        return name in self.samples

    def __len__(self) -> int:
        """Get number of loaded samples."""
        return len(self.samples)

    def __iter__(self):
        """Iterate over sample names."""
        return iter(self.samples)

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Uses scipy.signal.resample for accurate resampling.
        """
        if orig_sr == target_sr:
            return audio

        from scipy.signal import resample

        num_samples = int(len(audio) * target_sr / orig_sr)
        return resample(audio, num_samples).astype(np.float32)


class ClusterSampleManager(SampleManager):
    """
    Sample manager with cluster-based organization.

    Groups samples into clusters for random selection during sequencing.
    Extends SampleManager with cluster loading and random sample selection.

    Parameters
    ----------
    samples_dir : str or Path, optional
        Default directory for loading samples
    sample_rate : int
        Target sample rate for all samples (default: 44100)
    """

    def __init__(self, samples_dir: Optional[Union[str, Path]] = None,
                 sample_rate: int = 44100):
        super().__init__(samples_dir, sample_rate)
        self.clusters: Dict[str, List[str]] = {}

    def load_cluster_mapping(self, cluster_file: Union[str, Path],
                             samples_dir: Optional[Union[str, Path]] = None):
        """
        Load samples organized by cluster from a JSON mapping file.

        Parameters
        ----------
        cluster_file : str or Path
            Path to JSON file with cluster mappings
        samples_dir : str or Path, optional
            Directory containing samples (defaults to default_dir)

        Expected JSON format:
        {
            "cluster_0": ["sample1.wav", "sample2.wav"],
            "cluster_1": ["sample3.wav", "sample4.wav"]
        }
        """
        import json

        samples_dir = Path(samples_dir) if samples_dir else self.default_dir

        with open(cluster_file, 'r') as f:
            cluster_map = json.load(f)

        for cluster_id, sample_paths in cluster_map.items():
            self.clusters[cluster_id] = []
            for path in sample_paths:
                full_path = samples_dir / os.path.basename(path)
                name = f"{cluster_id}_{Path(path).stem.lower()}"
                try:
                    self.load(full_path, name)
                    self.clusters[cluster_id].append(name)
                except Exception as e:
                    print(f"Warning: Failed to load {full_path} for cluster {cluster_id}: {e}")

    def get_random_from_cluster(self, cluster_id: str) -> np.ndarray:
        """
        Get a random sample from a cluster.

        Parameters
        ----------
        cluster_id : str
            Cluster identifier

        Returns
        -------
        np.ndarray
            Audio data from a randomly selected sample
        """
        import random

        if cluster_id not in self.clusters:
            raise KeyError(f"Cluster '{cluster_id}' not found. Available: {list(self.clusters.keys())}")

        if not self.clusters[cluster_id]:
            raise ValueError(f"Cluster '{cluster_id}' is empty")

        sample_name = random.choice(self.clusters[cluster_id])
        return self.get(sample_name)

    def add_to_cluster(self, cluster_id: str, name: str,
                       audio_or_path: Union[np.ndarray, str],
                       sr: Optional[int] = None):
        """
        Add a sample to a cluster.

        Parameters
        ----------
        cluster_id : str
            Cluster identifier
        name : str
            Sample name
        audio_or_path : np.ndarray or str
            Audio data or file path
        sr : int, optional
            Sample rate (if audio_or_path is np.ndarray)
        """
        if cluster_id not in self.clusters:
            self.clusters[cluster_id] = []

        if isinstance(audio_or_path, np.ndarray):
            self.add(name, audio_or_path, sr)
        else:
            self.load(audio_or_path, name)

        self.clusters[cluster_id].append(name)

    def list_clusters(self) -> Dict[str, List[str]]:
        """Get all clusters and their sample names."""
        return dict(self.clusters)
