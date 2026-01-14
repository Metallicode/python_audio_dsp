"""
Audio effects and processing modules.

Core effects (no optional dependencies):
    from audio_dsp.effects import filter_effect, fuzz_distortion

Effects requiring librosa (install with: pip install audio-dsp[full]):
    from audio_dsp.effects import vocoder, autotune_effect, reverb_effect
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
from .negative_audio import create_negative_waveform, sidechain_compressor

__all__ = [
    # Core filters and distortion
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
    # Core effects
    "create_negative_waveform",
    "sidechain_compressor",
]

# Optional effects requiring librosa
try:
    from .tape_saturation import tape_saturation
    from .phaser import phaser_flanger_effect
    from .pitch_drift import pitch_drift
    from .pitch_flutter import flutter_effect
    from .super_delay import SuperDelay
    from .super_clean_compressor import SuperCleanCompressor
    from .vocoder import vocoder
    from .auto_tune import autotune_effect
    from .convolution_reverb import reverb_effect
    from .lofi import lofi_effect
    from .glitch import glitch_machine
    from .random_chorus import random_chorus
    from .temporal_gravity_warp import temporal_gravity_warp
    from .sitar_sympathetic_resonance import sitar_sympathetic_resonance
    from .variable_quantizer import variable_quantizer_effect
    from .melt_spectrum import melt_spectrum
    from .time_slice import time_slice_to_sines
    from .frequency_splicer import create_morphed_signal, splice_spectrum
    from .spectral_quantization import quantize_spectrum_stft_adaptive
    from .spectral_flow_compressor import spectral_flow_compressor
    from .topological_dynamics_compressor import topological_compressor
    from .fractional_calculus_compressor import fractional_compressor
    from .multi_band_saturation import process_multi_band

    __all__.extend([
        "tape_saturation",
        "phaser_flanger_effect",
        "pitch_drift",
        "flutter_effect",
        "SuperDelay",
        "SuperCleanCompressor",
        "vocoder",
        "autotune_effect",
        "reverb_effect",
        "lofi_effect",
        "glitch_machine",
        "random_chorus",
        "temporal_gravity_warp",
        "sitar_sympathetic_resonance",
        "variable_quantizer_effect",
        "melt_spectrum",
        "time_slice_to_sines",
        "create_morphed_signal",
        "splice_spectrum",
        "quantize_spectrum_stft_adaptive",
        "spectral_flow_compressor",
        "topological_compressor",
        "fractional_compressor",
        "process_multi_band",
    ])
except (ImportError, Exception):
    pass  # librosa not installed or broken

# Optional effects requiring opencv
try:
    from .audio_to_image_effects import audio_to_image, image_to_audio, apply_visual_effects

    __all__.extend([
        "audio_to_image",
        "image_to_audio",
        "apply_visual_effects",
    ])
except (ImportError, Exception):
    pass  # opencv not installed or broken
