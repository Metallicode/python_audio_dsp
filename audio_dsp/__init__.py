"""
audio-dsp: Python audio DSP library for synthesis, effects, and sequencing.

Submodules are imported lazily to avoid requiring all optional dependencies.
Import specific modules as needed:
    from audio_dsp.synth import SubtractiveSynth
    from audio_dsp.effects import filter_effect
"""

__version__ = "0.1.3"

__all__ = ["synth", "effects", "sequencer", "midi", "utils", "__version__"]


def __getattr__(name):
    """Lazy import submodules."""
    if name == "synth":
        from audio_dsp import synth
        return synth
    elif name == "effects":
        from audio_dsp import effects
        return effects
    elif name == "sequencer":
        from audio_dsp import sequencer
        return sequencer
    elif name == "midi":
        from audio_dsp import midi
        return midi
    elif name == "utils":
        from audio_dsp import utils
        return utils
    raise AttributeError(f"module 'audio_dsp' has no attribute '{name}'")
