# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
# Install in development mode
pip install -e .

# Install with all optional dependencies
pip install -e ".[full]"

# Install specific extras
pip install -e ".[synth]"   # Synthesis modules
pip install -e ".[midi]"    # MIDI processing (mido)
pip install -e ".[ml]"      # Machine learning (scikit-learn, umap)
pip install -e ".[viz]"     # Visualization (matplotlib)
pip install -e ".[audio]"   # Extended audio (librosa, pydub)

# Run tests
python -m pytest tests/

# Run a single test file
python tests/t_subtractive_synth.py
```

## Architecture

**Package structure:** `audio_dsp` with five main submodules that use lazy imports to avoid requiring all optional dependencies:

- **`synth`** - Synthesizers (SubtractiveSynth, DX7FMSynth, DrumSynth, chip tones, Karplus-Strong)
- **`effects`** - Audio effects with unified API: `effect(signal, sample_rate, **params) → numpy.ndarray`
- **`sequencer`** - Algorithmic composition (raga generator, tree/matrix composers, Game of Life sequencers)
- **`midi`** - MIDI utilities requiring mido (polyrhythmic generation, looping, alternate tunings)
- **`utils`** - Audio I/O, scales, noise algorithms, spectral analysis

**Core dependencies:** numpy, scipy, soundfile (always available)

**Optional dependencies:** librosa, mido, matplotlib, scikit-learn, opencv-python, simpleaudio

## Key Patterns

### Effects unified API
All effects follow: `effect_name(signal, sample_rate, **params) -> numpy.ndarray`
```python
from audio_dsp.effects import filter, delay, reverb
audio = filter(audio, sr, cutoff=1000, filter_type="lowpass")
audio = delay(audio, sr, delay_time=0.25, feedback=0.5)
```

### Backward compatibility aliases
Effects module exports both new names (`delay`, `compress`, `autotune`) and legacy aliases (`SuperDelay`, `SuperCleanCompressor`, `autotune_effect`).

### Synthesizer pattern
Synths are classes with configurable properties and a `synthesize(freq, duration, **envelope)` method:
```python
synth = SubtractiveSynth(sample_rate=44100)
synth.osc_wave = "saw"
synth.filter_cutoff = 800
audio = synth.synthesize(freq=220, duration=2.0)
```

### Audio I/O
`load_audio()` returns `(sample_rate, audio_array)` - note the scipy convention order.
```python
from audio_dsp.utils import load_audio
sr, audio = load_audio("file.wav", mono=True)
```

### Try/except import pattern
Submodules wrap optional dependencies in try/except blocks. When adding features requiring optional deps, follow this pattern in `__init__.py` files.
