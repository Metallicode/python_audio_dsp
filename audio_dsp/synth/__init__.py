"""
Synthesizer modules for audio generation.

Core synths (no optional dependencies):
    from audio_dsp.synth import SubtractiveSynth, DX7FMSynth, DrumSynth

Additional synths:
    from audio_dsp.synth import karplus_strong, generate_speech_synth
"""

from .subtractive_synth import SubtractiveSynth
from .dx7_fm_synth import DX7FMSynth
from .super_stacked_synth import SuperStackedSynth
from .drum_synth import DrumSynth
from .pluck import karplus_strong, generate_string_pluck
from .dialup_synth import generate_56k_dialup
from .speech import generate_speech_synth
from .chip_tone import (
    generate_kick,
    generate_snare,
    generate_cymbal,
    generate_blip,
    generate_swoosh,
    generate_fast_arp,
    generate_random_arp,
    generate_pop,
)

__all__ = [
    "SubtractiveSynth",
    "DX7FMSynth",
    "SuperStackedSynth",
    "DrumSynth",
    "karplus_strong",
    "generate_string_pluck",
    "generate_56k_dialup",
    "generate_speech_synth",
    "generate_kick",
    "generate_snare",
    "generate_cymbal",
    "generate_blip",
    "generate_swoosh",
    "generate_fast_arp",
    "generate_random_arp",
    "generate_pop",
]

# Optional synth requiring soundfile
try:
    from .physical_modeling_synth import PhysicalModelingSynth
    __all__.append("PhysicalModelingSynth")
except (ImportError, Exception):
    pass  # soundfile not installed or broken
