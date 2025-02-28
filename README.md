
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


Glad you’re digging the sound—it’s got that intriguing, organic vibe now with the transient exciting the spectral body! Let’s update the README to reflect this enhanced `PhysicalModelingSynth` with its amplitude-modulated excitation approach. I’ll tweak the previous description to highlight the new fusion method, keeping it punchy and technical while showcasing the suite’s creative potential.





# Physical Modeling Synthesis Suite

This Python-powered trio—`SpectralAnalyzer`, `TransientExtractor`, and `PhysicalModelingSynth`—unlocks a unique physical modeling synthesis workflow. By dissecting short WAV files into transients and spectral composites, then fusing them with a dynamic excitation model, it crafts sounds that feel alive—blending the raw snap of a pluck or hit with resonant, pitch-shifted bodies. Think gritty, evocative tones—perfect for experimental beats, cinematic soundscapes, or retro synth twists with a modern edge.

## Components

### SpectralAnalyzer
Captures the harmonic DNA of a sound, extracting its loudest spectral peaks.

- **Input**: Short WAV file (e.g., 0.1–1.0s—plucks, hits, clicks).
- **Process**: 2048-point FFT isolates the top 10 frequency components (frequency, amplitude, phase)—the sound’s resonant "body."
- **Output**: `.spectral` file (JSON)—spectral blueprint for resynthesis.
- **Usage**: Distill wood creaks, metal pings, or vocal bursts into harmonic seeds.

### TransientExtractor
Grabs the sharp, fleeting attack—the transient—that defines a sound’s onset.

- **Input**: WAV file.
- **Process**: RMS energy detection (256-sample windows) pinpoints the initial burst, up to 0.1s, with tunable threshold (default 0.01).
- **Output**: `.transient` file (JSON)—raw attack samples.
- **Usage**: Snag the crack of a drum, pluck of a string—pure percussive bite.

### PhysicalModelingSynth
Fuses transient and spectral data into a living sound, driven by physical modeling principles.

- **Inputs**:
  - `frequency`: Target pitch (Hz, e.g., 20–1000).
  - `length`: Output duration (seconds).
  - `transient_file`: `.transient` file—attack trigger.
  - `spectral_file`: `.spectral` file—resonant body.
  - `decay_rate`: Body decay speed (seconds, default 0.5)—controls sustain fade.
- **Process**: Transient excites the spectral body using amplitude modulation—its raw energy "strikes" the resonance, which rings out and decays naturally, pitch-shifted to `frequency` and stretched to `length`.
- **Output**: WAV file—transient snap ignites a tonal resonance, seamlessly infused.
- **Usage**: Transform a snare snap into a deep bass drone or a pluck into a shimmering chime—all with organic flow.

## Installation
Requires Python 3.8+ and these libraries:
```bash
pip install numpy librosa soundfile
```
- `numpy`: Signal crunching core.
- `librosa`: Audio analysis (FFT, RMS).
- `soundfile`: WAV file handling.

## Usage
### Workflow
1. **Extract Spectral Peaks**:
   ```python
   from spectral_analyzer import SpectralAnalyzer
   analyzer = SpectralAnalyzer()
   analyzer.analyze("input.wav", "input.spectral", num_peaks=10)
   ```
2. **Capture Transient**:
   ```python
   from transient_extractor import TransientExtractor
   extractor = TransientExtractor()
   extractor.extract("input.wav", "input.transient", threshold=0.01)
   ```
3. **Synthesize Sound**:
   ```python
   from physical_modeling_synth import PhysicalModelingSynth
   synth = PhysicalModelingSynth()
   synth.synthesize(frequency=110, length=2.0, transient_file="input.transient", 
                    spectral_file="input.spectral", output_file="pm_synth.wav", decay_rate=0.5)
   ```

### Customization
- **Spectral Richness**: Bump `num_peaks` in `SpectralAnalyzer`—e.g., 20 for denser resonance.
- **Transient Sensitivity**: Lower `threshold` (e.g., 0.001) in `TransientExtractor`—catches faint attacks.
- **Pitch & Decay**: Adjust `frequency` (e.g., 220 Hz), `length` (e.g., 4.0s), `decay_rate` (e.g., 0.2 for snappy, 1.0 for long) in `PhysicalModelingSynth`.

### Example
```python
analyzer = SpectralAnalyzer()
extractor = TransientExtractor()
synth = PhysicalModelingSynth()

analyzer.analyze("pluck.wav", "pluck.spectral")
extractor.extract("pluck.wav", "pluck.transient")
synth.synthesize(frequency=220, length=3.0, transient_file="pluck.transient", 
                 spectral_file="pluck.spectral", output_file="pluck_synth.wav", decay_rate=0.3)
```

## Sound Characteristics
- **Transient**: Raw, unfiltered attack—kicks off the sound with its original snap.
- **Body**: Spectral peaks resynthesized—excited by the transient’s energy, rings out at the target pitch with a natural decay.
- **Fusion**: Transient drives the body via amplitude modulation—no gaps, a seamless blend of attack and resonance.

## Notes
- **Input Files**: Best with short WAVs (0.1–1.0s)—e.g., plucks, hits, or sharp noises.
- **Harshness**: Crank `num_peaks` or `decay_rate` for wilder tones—gritty resonance awaits.
- **Flexibility**: Mix transients and spectra from different sources—drum attack with a bell’s ring!

## Limitations
- Batch synthesis only—real-time use needs optimization.
- Sine-based resynthesis—could extend to other waveforms.
- Transient capped at 0.1s—adjustable via `max_transient_len` in `TransientExtractor`.






# SuperStackedSynth: Ridiculously Over-Stacked Synthesizer

`SuperStackedSynth` is a Python-based synthesizer that takes basic waveforms—sine, triangle, saw, and square—and stacks them into a colossal, phasing wall of sound. With 1 to 10,000 slightly detuned oscillators, it crafts complex, swirling textures that evolve from lush pads to gritty, pulsating roars. Designed for sonic excess, it balances massive stacks with smart fade-ins and normalization—delivering consistent volume without the initial spike, all while keeping that raw, stacked vibe. Perfect for drones, monstrous leads, or just flexing synth absurdity.

## Features
- **Wave Shapes**: Sine, triangle, saw, square—mixable in any proportion for tonal variety.
- **Stacking**: 1 to 10,000 oscillators—each detuned for phasing, chorus-like richness.
- **Detuning**: Random spread (e.g., ±2% of base frequency)—creates beating, evolving textures.
- **Fade-In**: Per-oscillator fade (default 0.1s)—smooths the start, keeps volume steady from attack to sustain.
- **Normalization**: Scales amplitudes dynamically—handles absurd stacks without clipping.

## Installation
Requires Python 3.8+ and minimal dependencies:
```bash
pip install numpy soundfile
```
- `numpy`: Waveform generation and stacking.
- `soundfile`: WAV file output.

## Usage
### Basic Example
```python
from super_stacked_synth import SuperStackedSynth  # Assuming saved as super_stacked_synth.py
import soundfile as sf

# Initialize synth
synth = SuperStackedSynth(sample_rate=44100)

# Synthesize a thick pad with 100 oscillators
wave = synth.synthesize(base_freq=110, duration=4.0, num_oscillators=100, detune_spread=0.02)

# Save to WAV
sf.write("super_synth_100.wav", wave, 44100, subtype='PCM_16')
print("Saved to super_synth_100.wav")
```

### Customization
- **Oscillator Count**: Stack from subtle (1) to ridiculous (10,000).
  ```python
  synth.synthesize(base_freq=220, duration=4.0, num_oscillators=5000)
  ```
- **Wave Mix**: Blend shapes—e.g., saw-heavy for grit.
  ```python
  synth.synthesize(base_freq=220, duration=4.0, num_oscillators=5000, 
                   wave_mix={"sine": 0.1, "triangle": 0.1, "saw": 0.7, "square": 0.1})
  ```
- **Detuning**: Widen spread for more phasing (e.g., 0.05 = ±5%).
  ```python
  synth.synthesize(base_freq=440, duration=4.0, num_oscillators=10000, detune_spread=0.05)
  ```
- **Fade-In**: Adjust ramp time—short (0.05s) or long (0.5s).
  ```python
  synth.synthesize(base_freq=110, duration=4.0, num_oscillators=1000, fade_in_time=0.2)
  ```

### Full Stack Madness
```python
synth = SuperStackedSynth()
synth.synthesize(base_freq=440, duration=4.0, num_oscillators=10000, detune_spread=0.03, 
                 wave_mix={"square": 1.0}, fade_in_time=0.3, output_file="super_synth_max.wav")
```

## Parameters
- **base_freq**: Central frequency (Hz, e.g., 20–1000).
- **duration**: Sound length (seconds).
- **num_oscillators**: Stack size (1–10000)—more = denser texture.
- **detune_spread**: Max detuning factor (e.g., 0.02 = ±2%)—controls phasing width.
- **wave_mix**: Dict of weights (e.g., `{"sine": 0.5, "saw": 0.5}`)—shapes timbre.
- **fade_in_time**: Fade-in duration (seconds, e.g., 0.1)—smooths volume ramp.
- **output_file**: WAV output path.

## Sound Characteristics
- **Texture**: Phasing, beating richness—grows with oscillator count.
- **Volume**: Consistent from start to finish—smart fade-in avoids initial blasts.
- **Harshness**: Saw or square stacks deliver gritty, pulsating walls—sine or triangle softens it.

## Notes
- **Stack Size**: 100 for lush pads, 5000+ for sonic chaos—10,000 pushes phasing to the limit.
- **Detuning**: Small spreads (0.01) for subtle chorus, large (0.1) for wild drift.
- **Performance**: Handles 10,000 oscs—batch mode, not real-time (yet!).

## Limitations
- Batch synthesis only—real-time needs optimization.
- Fixed waveforms—could extend to noise or custom shapes.
- No envelopes—pure stack focus (add ADSR if you want!).



# DrumSynth: Versatile Drum Machine Synthesizer

`DrumSynth` is a Python-based drum machine that generates killer drum sounds from scratch—kick, snare, cymbal, clap, rim, and tom—each packed with customizable punch and texture. Built on a foundation of sine, square, and FM synthesis, layered with noise (normal, random, pink, brown), it offers a rich palette for crafting everything from tight electronic beats to gritty acoustic vibes. With pitch sweeps, signal mixing, and filtering, this synth delivers pro-grade drums—perfect for music production, sound design, or just banging out some serious rhythms.

## Features
- **Drum Types**:
  - **Kick**: Booming pitch sweep—deep and punchy.
  - **Snare**: Snappy noise + tonal zing—crisp and versatile.
  - **Cymbal**: Shimmery FM + noise—metallic ring.
  - **Clap**: Sharp noise bursts + tonal slap—tight and snappy.
  - **Rim**: High-pitched click + tone—crisp and percussive.
  - **Tom**: Fat pitch sweep + subtle noise—resonant thud.
- **Signals**:
  - **Sine**: Smooth, classic tone.
  - **Square**: Punchy, harmonic-rich edge.
  - **FM**: Complex, metallic shimmer—adjustable depth.
  - **Noise**: Normal, random, pink, brown—layered texture.
- **Customization**:
  - Signal mix—blend sine, square, FM (e.g., "square:0.7, sine:0.3").
  - Pitch sweeps, decay envelopes, and filters—tweak every hit.
- **Processing**: High/low/bandpass filters—shape the tone.

## Installation
Requires Python 3.8+ and these libraries:
```bash
pip install numpy soundfile
```
- `numpy`: Signal synthesis and processing.
- `soundfile`: WAV file output.

## Usage
### Basic Example
```python
from drum_synth import DrumSynth  # Assuming saved as drum_synth.py
import soundfile as sf

# Initialize synth
synth = DrumSynth(sample_rate=44100)

# Generate a punchy kick
kick = synth.kick(length=0.5, max_pitch=1000, min_pitch=50, signal_mix="square:0.7, sine:0.3")
sf.write("kick.wav", kick, 44100, subtype='PCM_16')
```

### Customization
- **Snare with FM Snap**:
  ```python
  snare = synth.snare(length=0.4, high_pitch=800, low_pitch=250, signal_mix="fm:1.0")
  sf.write("snare.wav", snare, 44100, subtype='PCM_16')
  ```
- **Cymbal with Mixed Tone**:
  ```python
  cymbal = synth.cymbal(length=1.0, op_a_freq=4000, op_b_freq=400, signal_mix="fm:0.8, square:0.2")
  sf.write("cymbal.wav", cymbal, 44100, subtype='PCM_16')
  ```
- **Fat Tom Blend**:
  ```python
  tom = synth.tom(length=0.5, max_pitch=300, min_pitch=80, signal_mix="sine:0.5, square:0.5")
  sf.write("tom.wav", tom, 44100, subtype='PCM_16')
  ```

## Parameters
- **length**: Duration (seconds, e.g., 0.1–1.0).
- **max_pitch/min_pitch**: Frequency range for sweeps (Hz, e.g., 50–1000).
- **decay_factor**: Envelope decay speed (e.g., 10–50)—sharp to long.
- **signal_mix**: Waveform blend (e.g., "sine:0.5, square:0.5")—weights sum to 1.
- **mix**: Noise vs. tone balance (0–1)—more noise or tone.
- **cutoff**: Filter frequency (Hz)—shapes timbre.

## Sound Characteristics
- **Kick**: Deep sweep—square adds punch, sine smooths it.
- **Snare**: Crisp attack—FM zings, noise snaps.
- **Cymbal**: Metallic ring—FM depth, square bite.
- **Clap**: Tight bursts—square sharpens the slap.
- **Rim**: Sharp tick—FM clicks, square edges.
- **Tom**: Resonant thud—sine warms, square fattens.

## Notes
- **Versatility**: Mix signals—e.g., "fm:0.7, square:0.3" for wild kicks.
- **Harshness**: Square/FM crank up grit—soften with sine or noise.
- **Offline**: Batch synthesis—perfect for studio quality, not real-time (yet!).

## Limitations
- Mono output—stereo could be added.
- Basic filters—FFT-based, room for advanced DSP.
- No effects—distortion or reverb could step it up.




-----------------------------------------------------------------------

#Effects 


# SuperCleanCompressor: State-of-the-Art Audio Compressor

`SuperCleanCompressor` is a Python-based audio compression powerhouse designed to deliver unparalleled dynamic control with pristine sound quality. Built from scratch for offline processing, it leverages infinite lookahead and oversampling to achieve surgical precision and musicality—rivaling the legends like the 1176, LA-2A, and SSL G-Series. With two distinct modes—"vintage" for warm, character-rich compression and "transparent" for crystal-clear dynamics—this compressor offers award-winning versatility, from subtle glue to brick-wall limiting, all without real-time constraints compromising its excellence.

## Features
- **Modes**:
  - **Vintage**: Warm, musical compression—soft knee, adaptive release, subtle tube-like saturation.
  - **Transparent**: Clean, precise dynamics—hard knee, peak accuracy, no coloration.
- **Parameters**:
  - `input_gain`: Boost input level (dB, e.g., 0–20).
  - `threshold`: Compression trigger point (dB, e.g., -60 to 0).
  - `ratio`: Dynamic reduction strength (e.g., 2:1 to ∞:1).
  - `attack`: Time to engage (seconds, e.g., 0.001–0.05)—ultra-precise with lookahead.
  - `release`: Time to recover (seconds, e.g., 0.01–1.0)—adaptive in vintage mode.
  - `knee_width`: Transition softness (dB, e.g., 6.0 = ±3 dB)—smooth or sharp.
  - `output_gain`: Final level tweak (dB).
  - `limit`: Hard cap at 0 dBFS—optional brick-wall limiter.
- **Processing**:
  - 2x oversampling—eliminates aliasing, enhances high-end clarity.
  - Full-signal lookahead—perfect attack/release timing, zero artifacts.
- **Quality**: Award-winning design—clean, musical, versatile—built for studio-grade results.

## Installation
Requires Python 3.8+ and these libraries:
```bash
pip install numpy soundfile librosa
```
- `numpy`: Core signal processing.
- `soundfile`: WAV I/O.
- `librosa`: Resampling for flexible input rates.

## Usage
### Basic Example
```python
from super_clean_compressor import SuperCleanCompressor  # Assuming saved as super_clean_compressor.py
import soundfile as sf

# Initialize compressor
compressor = SuperCleanCompressor(sample_rate=44100)

# Compress with transparent mode
compressor.compress("input.wav", "compressed.wav", mode="transparent", 
                    input_gain=6.0, threshold=-24.0, ratio=4.0, attack=0.005, 
                    release=0.05, output_gain=0.0, limit=True)
```

### Customization
- **Vintage Warmth**: Smooth, gluey compression.
  ```python
  compressor.compress("input.wav", "vintage_comp.wav", mode="vintage", 
                      input_gain=10.0, threshold=-20.0, ratio=2.0, attack=0.01, 
                      release=0.2, knee_width=6.0, output_gain=0.0, limit=False)
  ```
- **Brick-Wall Limiting**: Extreme dynamics control.
  ```python
  compressor.compress("input.wav", "limited.wav", mode="transparent", 
                      input_gain=20.0, threshold=-40.0, ratio=40.0, attack=0.001, 
                      release=0.01, knee_width=0.0, output_gain=0.0, limit=True)
  ```
- **Subtle Mix Bus**: Gentle leveling.
  ```python
  compressor.compress("input.wav", "mix_bus.wav", mode="vintage", 
                      input_gain=3.0, threshold=-30.0, ratio=4.0, attack=0.02, 
                      release=0.5, knee_width=12.0, output_gain=0.0, limit=False)
  ```

## Parameters
- **input_file**: Source WAV file path.
- **output_file**: Compressed WAV output path.
- **mode**: "vintage" (warm) or "transparent" (clean).
- **input_gain**: Input boost (dB, e.g., 0–20).
- **threshold**: Compression threshold (dB, e.g., -60 to 0).
- **ratio**: Compression ratio (e.g., 2.0–40.0, or higher for limiting).
- **attack**: Attack time (seconds, e.g., 0.001–0.05)—fast and precise.
- **release**: Base release time (seconds, e.g., 0.01–1.0)—adaptive in vintage mode.
- **knee_width**: Knee softness (dB, e.g., 0–12)—sharp to smooth transition.
- **output_gain**: Output adjustment (dB, e.g., -6 to 6).
- **limit**: True/False—hard limit at 0 dBFS.

## Sound Characteristics
- **Vintage Mode**: Warm, musical—soft knee, adaptive release, and subtle tube-like saturation glue dynamics with character.
- **Transparent Mode**: Clean, surgical—hard knee, peak-precise reduction, and crystal clarity for modern polish.
- **Dynamics**: From gentle leveling to brick-wall limiting—handles drums, vocals, or full mixes with finesse.

## Notes
- **Offline Power**: Infinite lookahead and 2x oversampling—zero artifacts, unmatched precision.
- **Harsh Settings**: Crank ratio (40:1+) and low threshold (-40 dB)—squashes dynamics to a razor’s edge.
- **Versatility**: Vintage warmth for soul, transparent clarity for control—award-worthy range.

## Limitations
- Batch processing only—optimized for quality, not real-time (yet!).
- Mono output—stereo support could be added.
- No sidechain or parallel mix—future enhancements possible.
