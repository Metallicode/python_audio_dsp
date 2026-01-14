"""
Utility functions for audio processing.

Core utilities (no optional dependencies):
    from audio_dsp.utils import generate_maqam_frequencies, white_noise
    from audio_dsp.utils import load_audio, save_audio, normalize_audio, resample_audio

Utilities requiring optional dependencies:
    from audio_dsp.utils import SpectralAnalyzer  # requires librosa
    from audio_dsp.utils import image_to_rhythmic_audio  # requires PIL, cv2
"""

from .audio_io import load_audio, save_audio, normalize_audio, resample_audio
from .maqamat import generate_maqam_frequencies
from .scales_and_melody import (
    categorise_interval,
    generate_scale,
    sine_wave,
    apply_envelope,
)
from .blend_modes import blend_audio, blend_wav_files
from .noise_algorithms import (
    white_noise,
    pink_noise,
    brown_noise,
    blue_noise,
    violet_noise,
    gaussian_noise,
    uniform_random_noise,
    perlin_noise,
    simplex_noise,
    fractal_noise,
    granular_noise,
    random_walk_noise,
    chaotic_noise,
    markov_chain_noise,
    tonal_noise,
    stochastic_resonance_noise,
    cellular_automata_noise,
)
__all__ = [
    "load_audio",
    "save_audio",
    "normalize_audio",
    "resample_audio",
    "generate_maqam_frequencies",
    "categorise_interval",
    "generate_scale",
    "sine_wave",
    "apply_envelope",
    "blend_audio",
    "blend_wav_files",
    "white_noise",
    "pink_noise",
    "brown_noise",
    "blue_noise",
    "violet_noise",
    "gaussian_noise",
    "uniform_random_noise",
    "perlin_noise",
    "simplex_noise",
    "fractal_noise",
    "granular_noise",
    "random_walk_noise",
    "chaotic_noise",
    "markov_chain_noise",
    "tonal_noise",
    "stochastic_resonance_noise",
    "cellular_automata_noise",
]

# Optional utilities requiring librosa
try:
    from .spectral_analyzer import SpectralAnalyzer
    from .transient_extractor import TransientExtractor
    from .arpeggio import generate_arpeggio
    __all__.extend(["SpectralAnalyzer", "TransientExtractor", "generate_arpeggio"])
except (ImportError, Exception):
    pass  # librosa not installed or broken

# Optional utilities requiring PIL/cv2
try:
    from .image_to_audio import image_to_rhythmic_audio, load_custom_image, apply_image_effect
    __all__.extend(["image_to_rhythmic_audio", "load_custom_image", "apply_image_effect"])
except (ImportError, Exception):
    pass  # PIL/cv2 not installed or broken
