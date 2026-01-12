"""
Utility functions for audio processing.

Core utilities (no optional dependencies):
    from audio_dsp.utils import generate_maqam_frequencies

Utilities requiring optional dependencies:
    from audio_dsp.utils.spectral_analyzer import SpectralAnalyzer  # requires librosa
    from audio_dsp.utils.image_to_audio import image_to_rhythmic_audio  # requires PIL, cv2
"""

from .maqamat import generate_maqam_frequencies
from .scales_and_melody import (
    categorise_interval,
    generate_scale,
    sine_wave,
    apply_envelope,
)

__all__ = [
    "generate_maqam_frequencies",
    "categorise_interval",
    "generate_scale",
    "sine_wave",
    "apply_envelope",
]
