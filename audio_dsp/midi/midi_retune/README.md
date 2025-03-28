

# MIDI to WAV Converter with Custom Octave Tuning

This Python script converts a MIDI file to a WAV audio file, allowing you to retune the octave ratio (normally 2:1) with a tuning offset. Instead of complex soundfonts, it uses pure sine waves with an exponential decay for simplicity and reliability. You can tweak the octave ratio—e.g., `-0.5` changes it to `1.5:1`, flattening the pitch space—while preserving the MIDI’s timing and dynamics.

## Features
- **MIDI Input**: Reads any standard `.mid` file, extracting note events (pitch, velocity, duration).
- **Custom Tuning**: Adjusts the octave ratio with a `tuning_offset` (e.g., `-0.5`, `+0.1`).
- **Simple Synthesis**: Generates sine waves with velocity-based amplitude and smooth decay.
- **Output**: Produces a WAV file at 44100 Hz, normalized to avoid clipping.

## Requirements
- **Python 3.x**
- **Libraries**: `numpy`, `scipy`, `mido` (`pip3 install numpy scipy mido`)
- **Input**: A MIDI file (`.mid`)

## Installation
1. **Install Python**: Ensure Python 3 is installed (`python3 --version`).
2. **Install Dependencies**:
   ```
   pip3 install numpy scipy mido
   ```
3. **Download Script**: Save this as `midi_retune_new.py`.

## Usage
1. **Prepare a MIDI File**: Place your `.mid` file (e.g., `test.mid`) in the same directory as the script.
2. **Run the Script**: Use the command line to convert the MIDI file to WAV with optional tuning.

### Basic Command
```
python3 midi_retune_new.py test.mid
```
- Output: `output.wav` with standard tuning (octave = 2:1).

### Custom Tuning Example
```
python3 midi_retune_new.py test.mid -o tuned.wav -t -0.5
```
- `-o tuned.wav`: Output file name.
- `-t -0.5`: Tuning offset, setting octave ratio to `2 - 0.5 = 1.5`.
- Result: A WAV file with a flatter pitch scale (e.g., C4 to C5 = 261.63 Hz to 392.45 Hz instead of 523.25 Hz).

### Command Options
- **`midi_file`**: Path to your MIDI file (required).
- **`--output, -o`**: Output WAV file name (default: `output.wav`).
- **`--tuning, -t`**: Tuning offset for octave ratio (default: `0.0`, range: e.g., `-1.0` to `+1.0`).

## How It Works

### MIDI Parsing
- **Library**: `mido` reads the MIDI file, extracting note-on/off events across all tracks.
- **Timing**: Converts MIDI ticks to seconds using `ticks_per_beat` and tempo (default 120 BPM = 500000 µs/beat).
- **Notes**: Stores pitch (MIDI note number), velocity (0-127), start time, and duration (minimum 10ms).

### Octave Retuning
- **Standard Tuning**: MIDI note `n` has frequency `freq = 440 * 2^((n - 69) / 12)`, where 69 = A4 (440 Hz), and 12 steps = 1 octave (2:1).
- **Custom Tuning**: With `tuning_offset`, the octave ratio becomes `octave_ratio = 2 + tuning_offset`. Frequency is:
  ```
  freq = 440 * (octave_ratio)^((n - 69) / 12)
  ```
  - Example: `-0.5` → `octave_ratio = 1.5`.
    - C4 (60): `440 * 1.5^(-9/12) ≈ 261.63 Hz`.
    - C5 (72): `440 * 1.5^(3/12) ≈ 392.45 Hz` (vs. 523.25 Hz normally).

### Synthesis
- **Waveform**: Sine wave with frequency from the retuned MIDI note, amplitude scaled by velocity (`velocity / 127`).
- **Decay**: Multiplies by `exp(-t * 3)` for a smooth fade-out, preventing clicks.
- **Mixing**: Notes are layered into a single audio array, starting at their MIDI-timed positions.

### Output
- **Normalization**: Scales the audio to a maximum amplitude of 1.
- **WAV**: Saved at 44100 Hz using `scipy.io.wavfile`.

## Theory
- **Octave Ratio**: Normally 2:1 (doubling frequency per 12 semitones). A negative `tuning_offset` (e.g., `-0.5`) compresses the pitch range, while a positive offset (e.g., `+0.1`) expands it.
- **Impact**: 
  - `-0.5`: Octave = 1.5:1, C4-C5 is ~130 Hz apart (not 261 Hz).
  - `+0.5`: Octave = 2.5:1, C4-C5 is ~349 Hz apart.
- **Musical Effect**: Alters the harmonic feel—flatter tunings sound moody or dense, sharper ones feel bright or stretched.

## Example
- **Input**: `test.mid` with C4 (60) to C5 (72), 1s each.
- **Command**: `python3 midi_retune_new.py test.mid -o tuned.wav -t -0.5`.
- **Output**: `tuned.wav`, 2s long:
  - C4 = 261.63 Hz.
  - C5 = 392.45 Hz (1.5:1 ratio).

## Troubleshooting
- **Small/Noisy Output**:
  - Check console: Look at `Total notes` and `duration`. Few notes or short duration might be the MIDI file.
  - Noise: Overlapping notes can interfere—try a simpler MIDI or adjust decay (`-t * 3` to `-t * 1`).
- **No Sound**: Ensure `test.mid` has valid note events (not just meta data).
- **Errors**: Verify dependencies (`numpy`, `scipy`, `mido`) are installed.

## Customization
- **Tuning**: Experiment with `-t` (e.g., `-0.1`, `+0.2`) for subtle or extreme effects.
- **Waveform**: Edit `synthesize_note` to use square (`np.sign(np.sin(...))`) or sawtooth waves.
- **Decay**: Change `-t * 3` to `-t * 1` for longer sustain.

