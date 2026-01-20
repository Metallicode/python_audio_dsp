[![PyPI Downloads](https://static.pepy.tech/personalized-badge/audio-dsp?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/audio-dsp)

# audio-dsp

A Python audio DSP library for synthesis, effects, and sequencing.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Metallicode/python_audio_dsp/blob/master/examples/Quickstart.ipynb)

## Installation

```bash
# Core installation (numpy, scipy, soundfile)
pip install audio-dsp

# Full installation with all optional dependencies
pip install audio-dsp[full]

# Specific extras
pip install audio-dsp[synth]   # Synthesis modules
pip install audio-dsp[midi]    # MIDI processing
pip install audio-dsp[ml]      # Machine learning features
pip install audio-dsp[viz]     # Visualization
pip install audio-dsp[audio]   # Extended audio processing (librosa, pydub)
```

## Features

### Synthesizers (`audio_dsp.synth`)
- **SubtractiveSynth** - Classic subtractive synthesis with oscillators, filters, LFO, ADSR
- **DX7FMSynth** - DX7-style FM synthesis with 4 operators and 5 algorithms
- **PhysicalModelingSynth** - Physical modeling synthesis
- **DrumSynth** - Drum synthesis
- **ChipTone functions** - Retro 8-bit chip synthesis (kick, snare, blip, etc.)
- **PluckSynth** - Karplus-Strong plucked string synthesis
- **DialupSynth** - Modem/dial-up sound synthesis

### Effects (`audio_dsp.effects`)
All effects follow a unified API: `effect(signal, sample_rate, **params) → numpy.ndarray`

- **Dynamics**: `compress`, multi-band saturation, sidechain compressor
- **Filters**: `filter` (lowpass, bandpass)
- **Modulation**: `phaser_flanger`, `chorus`, `flutter`, pitch drift
- **Time-based**: `delay`, `reverb` (convolution), time slice
- **Spectral**: Frequency splicer, melt spectrum, `vocoder`, spectral quantization
- **Character**: Distortion (19 types), `lofi`, `glitch`, tape saturation
- **Specialized**: `autotune`, sitar sympathetic resonance, `variable_quantizer`

### Sequencers (`audio_dsp.sequencer`)

**Unified API** - All sequencers share a common interface:
- `BaseSequencer` - Base class with `generate()`, `export()` methods
- `PatternSequencer` - Pattern-based sequencing with samples
- `LiquidSequencer` - Non-linear timing with swing/jitter
- `SampleManager` - Load samples from files or numpy arrays

**Specialized Sequencers:**
- `RagaSequencer` - Indian classical music with time-of-day raga selection
- `TreeSequencer` - Tree traversal-based algorithmic composition
- `ChordProgressionSequencer` - Microtonal chord progressions (24-TET, etc.)
- `MelodySequencer` - Melody development with counterpoint generation
- Game of Life sequencers with video output
- Matrix composer for linear algebra sonification

### MIDI (`audio_dsp.midi`)
- Polyrhythmic MIDI generation
- MIDI file looping
- Alternate tuning systems
- Logarithmic tunings

### Utilities (`audio_dsp.utils`)
- **Audio I/O** - Lightweight audio loading/saving with built-in WAV support (no external dependencies)
- Scale and melody utilities
- Spectral analysis
- Transient extraction
- Maqamat (Arabic scales)
- Noise algorithms
- Image to audio conversion


## Quick Start


| Notebook | Description | Link |
| :--- | :--- | :--- |
| **00. Quickstart** | Generate your first sound in 30 seconds. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Metallicode/python_audio_dsp/blob/master/examples/Quickstart.ipynb) |
| **01. Synthesizers** | Deep dive into Oscillators, FM, and Envelopes. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Metallicode/python_audio_dsp/blob/master/examples/Synthesis.ipynb) |
| **02. Effects Rack** | How to chain Reverb, Delay, and Distortion. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Metallicode/python_audio_dsp/blob/master/examples/Effects.ipynb) |
| **03. Sequencing** | creating generative melodies and MIDI tools. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Metallicode/python_audio_dsp/blob/master/examples/Sequencing.ipynb) |
| **04. Utilities** | Audio file handling and analysis tools. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Metallicode/python_audio_dsp/blob/master/examples/Utilities.ipynb) |



```python
from audio_dsp.synth import SubtractiveSynth
import soundfile as sf

# Create a subtractive synth
synth = SubtractiveSynth(sample_rate=44100)
synth.osc_wave = "saw"
synth.filter_cutoff = 800
synth.filter_resonance = 2.0

# Synthesize a note
audio = synth.synthesize(freq=220, duration=2.0)

# Save to file
sf.write("output.wav", audio, 44100)
```

```python
from audio_dsp.utils import load_audio
from audio_dsp.utils import wav_io
from audio_dsp.effects import filter, delay, reverb, compress

# Load audio
sr, audio = load_audio("input.wav", mono=True)

# Chain effects - all work on numpy arrays
audio = filter(audio, sr, cutoff=1000, filter_type="lowpass")
audio = delay(audio, sr, delay_time=0.25, feedback=0.5, mode="analog")
audio = reverb(audio, "impulse.wav", sr, wet_mix=0.3)
audio = compress(audio, sr, threshold=-20, ratio=4.0)

# Save output
wav_io.write("output.wav", sr, audio.astype('float32'))
```

```python
from audio_dsp.sequencer import PatternSequencer, SampleManager
from audio_dsp.synth import SubtractiveSynth
import numpy as np

# Create samples from synth output
synth = SubtractiveSynth()
kick = synth.synthesize(freq=60, duration=0.1)
snare = synth.synthesize(freq=200, duration=0.1)

# Create pattern sequencer
seq = PatternSequencer(bpm=120)
seq.add_sample("kick", kick)    # numpy array
seq.add_sample("snare", snare)
seq.add_sample("hihat", "samples/hihat.wav")  # or file path

# Generate from patterns
patterns = {
    "kick": "1000100010001000",
    "snare": "0000100000001000",
    "hihat": "1010101010101010"
}
audio = seq.generate_from_patterns(patterns, duration=4.0)
```

```python
from audio_dsp.sequencer import RagaSequencer

# Generate raga-based music appropriate for time of day
seq = RagaSequencer(bpm=90, root_frequency=220.0)
raga = seq.choose_raga()  # Selects based on current hour
print(f"Playing {raga['name']}")

audio = seq.generate_raga_phrase(raga, duration=30.0)
```

## License

MIT License - see [LICENSE](LICENSE) for details.
