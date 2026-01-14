"""
Music sequencing and algorithmic composition modules.

Import specific modules as needed:
    from audio_dsp.sequencer import play_music_with_fractal_rhythm
    from audio_dsp.sequencer import rule_based_tree_composer
"""

__all__ = []

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
