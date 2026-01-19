"""
Audio effects and processing modules.

Core effects (no optional dependencies):
    from audio_dsp.effects import filter, fuzz_distortion

Effects requiring librosa (install with: pip install audio-dsp[full]):
    from audio_dsp.effects import vocoder, autotune, reverb

Unified API pattern - all effects follow:
    effect_name(signal, sample_rate, **params) -> numpy.ndarray
"""

# Core effects that only require numpy/scipy
from .LP_BP_filter import filter, filter_effect  # filter_effect is backward compat alias
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
    # Core filters (new unified API)
    "filter",
    "filter_effect",  # backward compat
    # Distortion effects
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
    from .phaser import phaser_flanger, phaser_flanger_effect  # phaser_flanger_effect is backward compat
    from .pitch_drift import pitch_drift
    from .pitch_flutter import flutter, flutter_effect  # flutter_effect is backward compat
    from .super_delay import delay, SuperDelay  # SuperDelay is backward compat
    from .super_clean_compressor import compress, SuperCleanCompressor  # SuperCleanCompressor is backward compat
    from .vocoder import vocoder
    from .auto_tune import autotune, autotune_effect  # autotune_effect is backward compat
    from .convolution_reverb import reverb, reverb_effect  # reverb_effect is backward compat
    from .lofi import lofi, lofi_effect  # lofi_effect is backward compat
    from .glitch import glitch, glitch_machine  # glitch_machine is backward compat
    from .random_chorus import chorus, random_chorus  # random_chorus is backward compat
    from .temporal_gravity_warp import temporal_gravity_warp
    from .sitar_sympathetic_resonance import sitar_sympathetic_resonance
    from .variable_quantizer import variable_quantizer, variable_quantizer_effect  # variable_quantizer_effect is backward compat
    from .melt_spectrum import melt_spectrum
    from .time_slice import time_slice_to_sines
    from .frequency_splicer import create_morphed_signal, splice_spectrum
    from .spectral_quantization import quantize_spectrum_stft_adaptive
    from .spectral_flow_compressor import spectral_flow_compressor
    from .topological_dynamics_compressor import topological_compressor
    from .fractional_calculus_compressor import fractional_compressor
    from .multi_band_saturation import process_multi_band

    __all__.extend([
        # New unified API names
        "tape_saturation",
        "phaser_flanger",
        "pitch_drift",
        "flutter",
        "delay",
        "compress",
        "vocoder",
        "autotune",
        "reverb",
        "lofi",
        "glitch",
        "chorus",
        "temporal_gravity_warp",
        "sitar_sympathetic_resonance",
        "variable_quantizer",
        "melt_spectrum",
        "time_slice_to_sines",
        "create_morphed_signal",
        "splice_spectrum",
        "quantize_spectrum_stft_adaptive",
        "spectral_flow_compressor",
        "topological_compressor",
        "fractional_compressor",
        "process_multi_band",
        # Backward compatibility aliases
        "phaser_flanger_effect",
        "flutter_effect",
        "SuperDelay",
        "SuperCleanCompressor",
        "autotune_effect",
        "reverb_effect",
        "lofi_effect",
        "glitch_machine",
        "random_chorus",
        "variable_quantizer_effect",
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
