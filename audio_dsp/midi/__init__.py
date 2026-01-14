"""
MIDI processing and utilities.

Requires mido (install with: pip install audio-dsp[midi])

Import specific modules as needed:
    from audio_dsp.midi import generate_polyrhythmic_midi, midi_looper
"""

try:
    from .polyrhythmic_midi import generate_polyrhythmic_midi
    from .midi_looper import midi_looper

    __all__ = [
        "generate_polyrhythmic_midi",
        "midi_looper",
    ]
except (ImportError, Exception):
    __all__ = []  # mido not installed or broken
