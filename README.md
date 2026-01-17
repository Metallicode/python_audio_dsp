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
- **Dynamics**: Compressors, multi-band saturation, negative audio
- **Filters**: Low-pass, band-pass, high-pass filters
- **Modulation**: Phaser, chorus, pitch drift, pitch flutter
- **Time-based**: Delay, convolution reverb, time slice
- **Spectral**: Frequency splicer, melt spectrum, vocoder, spectral quantization
- **Character**: Distortion, lo-fi, glitch, tape saturation
- **Specialized**: Auto-tune, sitar sympathetic resonance

### Sequencers (`audio_dsp.sequencer`)
- Raga generator with Indian classical music scales
- Matrix and tree-based algorithmic composers
- Game of Life sequencers
- Text-based sequencing with sample clustering
- Chord progression generators
- Microtonal support

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
from audio_dsp.utils import load_audio, save_audio
from audio_dsp.effects import vocoder

# Load audio (accepts file paths or numpy arrays)
sr, carrier = load_audio("carrier.wav")
sr, modulator = load_audio("modulator.wav")

# Apply vocoder effect
output, sr = vocoder(carrier, modulator, sr, n_filters=16)

# Save output
save_audio("output.wav", output, sr)
```

## License

MIT License - see [LICENSE](LICENSE) for details.
