"""
Music sequencing and algorithmic composition modules.

This module provides a unified API for pattern-based and generative sequencers.

Unified API (recommended):
    from audio_dsp.sequencer import PatternSequencer, SampleManager
    from audio_dsp.sequencer import BaseSequencer, GenerativeSequencer

Legacy API (for backwards compatibility):
    from audio_dsp.sequencer import play_music_with_fractal_rhythm
    from audio_dsp.sequencer import rule_based_tree_composer

Example usage:
    >>> from audio_dsp.sequencer import PatternSequencer
    >>> seq = PatternSequencer(bpm=120)
    >>> seq.add_sample("kick", "samples/drums/kick.wav")
    >>> seq.add_sample("snare", synth.synthesize(200, 0.1))  # numpy array
    >>> patterns = {"kick": "1000100010001000", "snare": "0000100000001000"}
    >>> audio = seq.generate_from_patterns(patterns, duration=4.0)
"""

__all__ = []

# Core unified API (always available - no optional dependencies)
try:
    from .base import BaseSequencer, PatternBasedSequencer, GenerativeSequencer
    from .sample_manager import SampleManager, ClusterSampleManager, get_default_samples_dir
    from .pattern_sequencer import PatternSequencer, LiquidSequencer
    __all__.extend([
        "BaseSequencer", "PatternBasedSequencer", "GenerativeSequencer",
        "SampleManager", "ClusterSampleManager", "get_default_samples_dir",
        "PatternSequencer", "LiquidSequencer"
    ])
except (ImportError, Exception):
    pass

# Raga generator (requires simpleaudio)
try:
    from .raga_generator import play_music_with_fractal_rhythm, generate_fractal_rhythm
    __all__.extend(["play_music_with_fractal_rhythm", "generate_fractal_rhythm"])
except (ImportError, Exception):
    pass

# Tree composer (core numpy/scipy)
try:
    from .tree_composer import rule_based_tree_composer, build_tree, traverse_tree
    __all__.extend(["rule_based_tree_composer", "build_tree", "traverse_tree"])
except (ImportError, Exception):
    pass

# Chord progressions (core numpy/scipy)
try:
    from .stepping_chord_progressions import generate_progression, generate_scale as generate_chord_scale
    __all__.extend(["generate_progression", "generate_chord_scale"])
except (ImportError, Exception):
    pass

# Melody development (core numpy/scipy)
try:
    from .melody_choice import develop_melody, generate_counterpoint
    __all__.extend(["develop_melody", "generate_counterpoint"])
except (ImportError, Exception):
    pass

# Matrix composer has pyaudio initialization at module level, skip auto-import
# Use: from audio_dsp.sequencer.matrix_composer import play_matrix
