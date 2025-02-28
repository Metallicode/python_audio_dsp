
# Python Audio DSP
Is a set of tools for the audio enthusiast, including:

`SubtractiveSynth` is a Python-based subtractive synthesizer designed to emulate the gritty, resonant basslines of the Roland TB-303, with flexible oscillator waveforms, a 4-pole analog-style filter, and a dynamic ADSR envelope. Built for audio DSP enthusiasts, it offers a harsh, squelchy sound with extensive parameter control, including pitch modulation via LFO, filter resonance, and Q width—all tweakable to craft everything from deep acid bass to sharp, cutting leads.

## Features
- **Oscillators**: Supports multiple waveforms:
  - Sine, square, sawtooth, triangle, pulse-width modulation (PWM), and custom wavetable.
- **Filter**: 4-pole (24 dB/oct) resonant filter with three types:
  - **Low-pass**: Classic TB-303 squelch—cuts highs, boosts resonance.
  - **High-pass**: Sharp low-frequency attenuation—ideal for crisp, edgy tones.
  - **Bandpass**: Mid-range focus—perfect for nasal, resonant sweeps.
  - Adjustable cutoff (50–5000 Hz), resonance (0–5), and Q factor (0.1–10) for peak width.
- **Envelope**: ADSR (Attack, Decay, Sustain, Release) for amplitude shaping—fast snaps to long decays.
- **LFO**: Low-frequency oscillator modulates filter cutoff, pitch, or amplitude—adds wobble or vibrato.
- **Harshness**: Multi-stage `tanh` saturation—gritty, analog-style distortion baked into the filter.

## Installation
Requires Python 3.8+ and the following libraries:
```bash
pip install numpy scipy librosa soundfile
```
- `numpy`: Signal processing.
- `scipy`: Filter utilities (though not used directly here—ladder filter is custom).
- `librosa`: Audio handling (waveform loading).
- `soundfile`: WAV file output.

## Usage
### Basic Example
```python
from subtractive_synth import SubtractiveSynth  # Assuming saved as subtractive_synth.py
import soundfile as sf

# Initialize synth
s = SubtractiveSynth(sample_rate=44100)

# Synthesize a TB-303-style bass note
wave = s.synthesize(freq=70, duration=4.0, attack=0.02, decay=0.1, sustain=0.6, release=0.3)

# Save to WAV
sf.write("tb303_bass.wav", wave, 44100, subtype='PCM_16')
print("Saved to tb303_bass.wav")
```

### Parameter Tweaks
- **Oscillator Type**: Change `s.osc_wave` to `"square"`, `"saw"`, etc.
  ```python
  s.osc_wave = "square"
  ```
- **Filter Settings**: Adjust `filter_type`, `filter_cutoff`, `filter_resonance`, `filter_q`.
  ```python
  s.filter_type = "highpass"  # Switch to high-pass
  s.filter_cutoff = 1500      # Higher cutoff
  s.filter_resonance = 3.0    # Intense resonance
  s.filter_q = 5.0            # Narrow peak
  ```
- **LFO**: Modulate pitch instead of filter.
  ```python
  s.lfo_target = "pitch"
  s.lfo_freq = 2.0
  s.lfo_depth = 0.3
  ```

### Full Customization
```python
s = SubtractiveSynth(sample_rate=44100)
s.osc_wave = "saw"
s.filter_type = "bandpass"
s.filter_cutoff = 800
s.filter_resonance = 4.0
s.filter_q = 2.0
s.lfo_target = "filter"
s.lfo_freq = 6.0
s.lfo_depth = 0.5
wave = s.synthesize(freq=110, duration=2.0, attack=0.01, decay=0.05, sustain=0.8, release=0.2)
sf.write("tb303_custom.wav", wave, 44100, subtype='PCM_16')
```

## Parameters
- **freq**: Oscillator frequency (Hz, e.g., 20–1000).
- **duration**: Sound length (seconds).
- **attack**: Envelope attack time (seconds, e.g., 0.01–1.0).
- **decay**: Envelope decay time (seconds, e.g., 0.05–1.0).
- **sustain**: Sustain level (0–1).
- **release**: Envelope release time (seconds, e.g., 0.1–1.0).
- **filter_cutoff**: Base cutoff frequency (Hz, 50–5000).
- **filter_resonance**: Resonance amplitude (0–5, 5 = max self-oscillation).
- **filter_q**: Q factor (0.1–10, lower = wider peak).
- **osc_wave**: Waveform type (see Features).
- **filter_type**: Filter mode ("lowpass", "highpass", "bandpass").
- **lfo_freq**: LFO frequency (Hz, e.g., 0.1–20).
- **lfo_depth**: LFO modulation depth (0–1).
- **lfo_target**: Modulation target ("filter", "pitch", "amplitude").

## Notes
- **TB-303 Vibe**: Set `osc_wave="saw"`, `filter_type="lowpass"`, high `filter_resonance` (3.0–5.0), and short `decay` (0.05–0.2) for that iconic acid bass.
- **Harshness**: Built-in `tanh` saturation adds grit—crank `filter_resonance` for squeals.
- **Performance**: 4-pole filter is stateful—suitable for real-time if optimized further.

## Limitations
- No glide/portamento—yet! Add it by interpolating `freq` over time in `synthesize`.
- Single oscillator—stack more via multiple calls for fatter sounds.
- Filter is analog-inspired—not a perfect 303 clone, but close with tweaking.

