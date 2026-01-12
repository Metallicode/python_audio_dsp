"""
Synthesizer modules for audio generation.

Core synths (no optional dependencies):
    from audio_dsp.synth import SubtractiveSynth, DX7FMSynth

Import specific modules for additional synths:
    from audio_dsp.synth.drum_synth import DrumSynth
    from audio_dsp.synth.pluck import karplus_strong
"""

from .subtractive_synth import SubtractiveSynth
from .dx7_fm_synth import DX7FMSynth
from .super_stacked_synth import SuperStackedSynth
from .drum_synth import DrumSynth

__all__ = [
    "SubtractiveSynth",
    "DX7FMSynth",
    "SuperStackedSynth",
    "DrumSynth",
]
