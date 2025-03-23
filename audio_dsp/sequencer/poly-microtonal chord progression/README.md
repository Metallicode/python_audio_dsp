# Poly-Microtonal Chord Progression Generator

This Python script generates a microtonal chord progression using multiple equal divisions of the octave (EDO) tuning systems. It combines traditional 12-EDO note names with flexible N-EDO tuning, allowing you to create rich, polyphonic progressions with custom scale degrees and chord qualities. The output is a WAV file based on a single-note sample (`sample.wav`) pitched to match the specified frequencies.

## Features
- **Flexible Tuning**: Supports any N-EDO system (e.g., 12, 19, 22, 24, 31, 43, 53).
- **Chord Types**: Major, minor, suspended (sus4), and dominant 7th chords.
- **Scale Degrees**: Select chord roots from degrees I-VII within each N-EDO system.
- **Customizable**: Define progressions with a simple string syntax, set BPM, and adjust voices.
- **Sample-Based**: Uses a 440 Hz (A4) sample as the base sound, pitch-shifted for all notes.

## Requirements
- **Python 3.x**
- **Libraries**: `numpy`, `scipy`, `librosa` (`pip install numpy scipy librosa`)
- **Input File**: `sample.wav` (a single-note audio file tuned to 440 Hz, A4)

## Usage
1. **Prepare Your Sample**: Place a `sample.wav` file (mono, 440 Hz) in the same directory as the script.
2. **Edit the Chord Progression**: Modify the `chord_str` in the `if __name__ == "__main__":` block to define your progression.
3. **Run the Script**: Execute `python microtonal_diverse_edo.py` in your terminal.
4. **Output**: A WAV file (`progression_diverse_edo.wav`) will be generated with your progression.

### Syntax
The chord progression is defined as a comma-separated string in the format:
```
"Note[quality]-degree-N-EDO-Duration, ..."
```
- **Note**: A 12-EDO note name (e.g., `C`, `A#`, `F`).
- **Quality**: Optional chord type (`m` for minor, `sus` for sus4, `7` for dominant 7th; omit for major).
- **Degree**: Roman numeral (I, II, III, IV, V, VI, VII) selecting the scale degree in N-EDO.
- **N-EDO**: Number of equal divisions per octave (e.g., `12`, `31`, `19`).
- **Duration**: Note length as a fraction (e.g., `1/4` for quarter note, `1/8` for eighth note).

**Example**:
```python
chord_str = "Cm-I-31-1/4, G-IV-19-1/4, A#-VII-12-1/4, F7-II-24-1/4"
```
- `Cm-I-31-1/4`: C minor chord, root degree I, 31-EDO, quarter note.
- `G-IV-19-1/4`: G major chord, 4th degree, 19-EDO, quarter note.

### Parameters
- **bpm**: Beats per minute (e.g., `100`). A quarter note (1/4) at 100 BPM is 0.15 seconds.
- **voices**: Number of notes per chord (default 3; set to 4 for full 7th chords).
- **output_file**: Name of the generated WAV file (default `progression_diverse_edo.wav`).

## Theory Behind the Script

### Note and Chord Selection
1. **Root Note (12-EDO Base)**:
   - The script starts with a 12-EDO note name (e.g., `C` = 261.63 Hz, `A#` = 466.16 Hz) relative to A4 = 440 Hz.
   - This root is calculated using a 12-EDO step map: `C = -9`, `A = 0`, `A# = 1`, etc.
   - Frequency: `base_freq = 440 * 2^(steps_12edo / 12)`.

2. **Conversion to N-EDO**:
   - The 12-EDO root frequency is converted to steps in the specified N-EDO system: `steps_from_a4 = n_edo * log2(base_freq / 440)`.
   - Example: `C` (261.63 Hz) in 31-EDO = `31 * log2(261.63/440) ≈ -14.26 steps`.

This expression calculates the number of steps in an N-EDO tuning system from the reference pitch A4 (440 Hz) to the root note’s frequency (base_freq), which is initially determined in 12-EDO. It’s the bridge between the familiar 12-tone system (where note names like C, G, A# live) and the microtonal N-EDO system (e.g., 31-EDO, 19-EDO) you specify for each chord.

n_edo: The number of equal divisions per octave (e.g., 12, 31, 19).
base_freq: The frequency of the root note in 12-EDO (e.g., C4 = 261.63 Hz).
440: The reference frequency of A4.
log2(base_freq / 440): The logarithmic distance (in octaves) from 440 Hz to base_freq.
n_edo * log2(base_freq / 440): Converts that distance into N-EDO steps.

***The Math Behind It***
Frequency in equal temperament is exponential: freq = ref_freq * 2^(steps / divisions). To find steps from a frequency, you reverse it with a logarithm.

Frequency Ratio:
base_freq / 440 is the ratio of the root note’s frequency to A4.
Example: G4 = 392 Hz, so 392 / 440 ≈ 0.8909.
Log Base 2:
log2(base_freq / 440) gives the number of octaves (positive or negative) from 440 Hz.
log2(0.8909) ≈ -0.1674 (G4 is ~0.1674 octaves below A4).
Scale to N-EDO:
Multiply by n_edo to get steps in that system.
For 19-EDO: 19 * -0.1674 ≈ -3.18 steps.
This result (steps_from_a4) is the root’s position relative to A4 in N-EDO, which we then adjust with the degree and chord intervals.



3. **Scale Degree Adjustment**:
   - The specified degree (I = 0, IV = 3, VII = 6) shifts the chord’s root within the N-EDO system: `root_steps = steps_from_a4 + degree`.
   - Example: `G-IV-19` shifts G (392 Hz, ~-2.67 steps in 19-EDO) by +3 steps to ~0.33 steps (~440 Hz).

4. **Chord Construction**:
   - Intervals are defined in 12-EDO terms and scaled to N-EDO:
     - Major: 0, 4, 7 steps.
     - Minor: 0, 3, 7 steps.
     - Sus4: 0, 5, 7 steps.
     - 7th: 0, 4, 7, 10 steps (4 voices needed for the 10th).
   - Scaling: `total_steps = root_steps + (step * n_edo / 12)`.
   - Frequency: `target_freq = 440 * 2^(total_steps / n_edo)`.

5. **Polyphony**:
   - Each chord stacks `voices` notes (default 3), normalizing amplitude by dividing by the number of voices.

### Microtonal Theory
- **EDO Systems**: N-EDO divides the octave (2:1 frequency ratio) into N equal steps. Step size = `2^(1/N)`.
  - 12-EDO: Semitone = `2^(1/12)` (~100 cents).
  - 31-EDO: Step = `2^(1/31)` (~38.7 cents).
  - 19-EDO: Step = `2^(1/19)` (~63.2 cents).
- **Why Diverse EDOs?**:
  - 12: Familiar Western tuning.
  - 19: Smooth thirds and fifths.
  - 22: Exotic, non-Western intervals.
  - 24: Quarter tones.
  - 31, 43, 53: Increasingly fine microtonality, approximating just intonation.
- **Harmony**: Combining EDOs creates a poly-microtonal texture—chords shift between familiar and alien sonorities.

## Example Progression
The default 32-chord progression uses an AABA form (8+8+8+8 bars) at 100 BPM:
- **EDOs**: 12, 19, 22, 24, 31, 43, 53.
- **Chords**: Mix of major, minor, sus, 7th with degrees I-VII.
- **Duration**: 19.2 seconds (32 quarter notes).

Run it to hear a journey from Cm in 31-EDO to exotic shifts in 22 and 53-EDO, grounded by 12-EDO pivots.

## Customization
- **Add EDOs**: Replace values in `chord_str` (e.g., try 17, 72).
- **Change Degrees**: Use I-VII to explore different scale positions.
- **Mix Durations**: Swap 1/4 for 1/8 or 1/2 for rhythmic variety.
- **Voices**: Set `voices=4` in `main()` for full 7th chords.

## Troubleshooting
- **“Blazing Fast” Output**: Check console for `Total duration`—should match BPM (e.g., 19.2s at 100 BPM).
- **Invalid Degree**: Use only I, II, III, IV, V, VI, VII—e.g., "IIV" errors out.
- **No Sound**: Ensure `sample.wav` exists and is 440 Hz.

Enjoy your microtonal adventures!