"""
Audio effects and processing modules.

Core effects (no optional dependencies):
    from audio_dsp.effects import filter_effect, fuzz_distortion

Effects requiring librosa (install with: pip install audio-dsp[full]):
    from audio_dsp.effects.vocoder import vocoder
    from audio_dsp.effects.auto_tune import autotune_effect
    from audio_dsp.effects.convolution_reverb import reverb_effect
"""

# Core effects that only require numpy/scipy
from .LP_BP_filter import filter_effect
from .distortion import (
    fuzz_distortion,
    overdrive_distortion,
    saturation_distortion,
    cubic_distortion,
    hard_clip_distortion,
    wavefold_distortion,
    bitcrush_distortion,
    asymmetric_distortion,
    logistic_distortion,
    poly_distortion,
    triangle_fold_distortion,
    sawtooth_fold_distortion,
    chebyshev_fold_distortion,
    parabolic_fold_distortion,
    exp_fold_distortion,
    fractal_fold_distortion,
    mirror_fold_distortion,
    dynamic_triangle_fold_distortion,
    frequency_lock_distortion,
)

__all__ = [
    "filter_effect",
    "fuzz_distortion",
    "overdrive_distortion",
    "saturation_distortion",
    "cubic_distortion",
    "hard_clip_distortion",
    "wavefold_distortion",
    "bitcrush_distortion",
    "asymmetric_distortion",
    "logistic_distortion",
    "poly_distortion",
    "triangle_fold_distortion",
    "sawtooth_fold_distortion",
    "chebyshev_fold_distortion",
    "parabolic_fold_distortion",
    "exp_fold_distortion",
    "fractal_fold_distortion",
    "mirror_fold_distortion",
    "dynamic_triangle_fold_distortion",
    "frequency_lock_distortion",
]
