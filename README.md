
# Python Audio DSP
Is a set of tools for the audio enthusiast, including:

# SubtractiveSynth: Classic Subtractive Synthesizer

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





# DX7FMSynth: Yamaha DX7-Inspired FM Synthesizer

`DX7FMSynth` is a Python-based frequency modulation (FM) synthesizer that channels the iconic sound of the Yamaha DX7, delivering those punchy, metallic, and glassy tones that defined 80s synth pop and beyond. With 4 operators, individual ADSR envelopes, operator feedback, and a selection of classic modulation architectures, this synth offers a gritty, versatile platform for crafting everything from deep acid basslines to shimmering bells—all with that unmistakable FM edge.

## Features
- **Operators**: 4 sine-wave oscillators, configurable as carriers or modulators, with frequency ratios and modulation indices for rich FM textures.
- **ADSR Envelopes**: Each operator has its own Attack, Decay, Sustain, and Release envelope—shape amplitude dynamics with precision (default: fast attack, short decay, strong sustain).
- **Feedback**: Operator 4 feeds back into itself (0–5 range)—adds that biting, metallic DX7 character.
- **Modulation Architectures**: 5 predefined algorithms inspired by the DX7:
  1. **Linear Stack**: 4 → 3 → 2 → 1—deep, evolving tones.
  2. **Dual Stack**: (4 → 3) + (2 → 1)—bright, layered mix.
  3. **Parallel Mod**: (4 + 3) → 2 → 1—complex, metallic resonance.
  4. **Multi-Carrier**: 4 → (3 + 2) → 1—rich, chime-like output.
  5. **Independent**: All operators summed—no modulation, pure additive with feedback.
- **Harshness**: Built-in FM modulation and feedback—push `mod_indices` or `feedback` for that raw, clangy DX7 bite.

## Installation
Requires Python 3.8+ and minimal dependencies:
```bash
pip install numpy soundfile
```
- `numpy`: Core signal generation.
- `soundfile`: WAV file output.

## Usage
### Basic Example
```python
from dx7_fm_synth import DX7FMSynth  # Assuming saved as dx7_fm_synth.py
import soundfile as sf

# Initialize synth
synth = DX7FMSynth(sample_rate=44100)

# Synthesize a classic DX7 bass
wave = synth.synthesize(freq=110, duration=2.0, algorithm=1)

# Save to WAV
sf.write("dx7_bass.wav", wave, 44100, subtype='PCM_16')
print("Saved to dx7_bass.wav")
```

### Customization
- **Algorithm**: Switch modulation routing (1–5).
  ```python
  synth.algorithm = 3  # Parallel mod architecture
  ```
- **Feedback**: Add grit to Op4.
  ```python
  synth.feedback = 2.0  # Moderate feedback
  ```
- **Modulation Indices**: Control FM intensity.
  ```python
  synth.mod_indices = [2.0, 1.5, 1.0, 0.5]  # Stronger mod on Op1
  ```
- **Frequency Ratios**: Tune operator relationships.
  ```python
  synth.freq_ratios = [1.0, 3.0, 2.0, 0.5]  # Harmonic richness
  ```
- **ADSR**: Shape each operator’s envelope.
  ```python
  synth.adsr[0] = {'attack': 0.01, 'decay': 0.05, 'sustain': 0.8, 'release': 0.3}  # Op1
  synth.adsr[3] = {'attack': 0.02, 'decay': 0.2, 'sustain': 0.5, 'release': 0.4}  # Op4
  ```

### Full Example
```python
synth = DX7FMSynth(sample_rate=44100)
synth.algorithm = 2  # Dual stack
synth.feedback = 3.0  # Harsh feedback
synth.mod_indices = [2.5, 1.8, 1.2, 0.8]
synth.freq_ratios = [1.0, 2.0, 4.0, 0.25]
synth.adsr[0]['decay'] = 0.1  # Tighten Op1 decay
wave = synth.synthesize(freq=220, duration=2.0)
sf.write("dx7_bell.wav", wave, 44100, subtype='PCM_16')
```

## Parameters
- **freq**: Base frequency (Hz, e.g., 20–1000).
- **duration**: Sound length (seconds).
- **algorithm**: Modulation routing (1–5, see Features).
- **freq_ratios**: Operator frequency ratios (list of 4 floats, e.g., [1.0, 2.0, 1.0, 0.5]).
- **mod_indices**: Modulation strength (list of 4 floats, e.g., 0.1–5.0).
- **feedback**: Op4 feedback amount (0–5).
- **adsr**: List of 4 dicts with `'attack'`, `'decay'`, `'sustain'`, `'release'` (seconds or 0–1 for sustain).

## Notes
- **DX7 Vibe**: Use `algorithm=1`, `freq=110`, high `mod_indices` (2.0–3.0), and `feedback=2.0` for deep FM bass. Try `algorithm=4`, `freq=440` for bell-like tones.
- **Harshness**: Crank `feedback` or `mod_indices`—gets wild fast!
- **Flexibility**: Tweak `freq_ratios` for harmonic or inharmonic timbres.

## Limitations
- No detuning—add by offsetting `freq_ratios` slightly (e.g., 1.01).
- Fixed sine waves—DX7 used sine only, but could extend to other shapes.
- No real-time—batch processing for now; optimize for live use if needed.
