import numpy as np
from audio_dsp.utils import wav_io as wavfile
import librosa
import re
from audio_dsp.utils import load_audio, resample_audio

SAMPLE_RATE = 44100

def save_wav(file_path, data):
    data = np.clip(data, -1, 1)
    wavfile.write(file_path, SAMPLE_RATE, (data * 32767).astype(np.int16))
    print(f"Saved to {file_path}")

def pitch_shift(sample, sr, target_freq, base_freq=440.0):
    rate = target_freq / base_freq
    shifted = librosa.effects.pitch_shift(sample, sr=sr, n_steps=np.log2(rate) * 12)
    return shifted

def parse_chord_string(chord_str):
    """Parse 'Cm-I-31-1/4, G-IV-16-1/4' into (note, quality, degree, edo, duration)."""
    chords = chord_str.split(", ")
    pattern = re.compile(r"([A-G]#?)(m|sus|7)?-([IV]+)-(\d+)-(\d+/\d+)")
    degree_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5, 'VII': 6}
    result = []
    for chord in chords:
        match = pattern.match(chord.strip())
        if match:
            note, quality, degree_str, edo, dur_str = match.groups()
            duration = eval(dur_str)
            quality = quality if quality else "major"
            if degree_str not in degree_map:
                raise ValueError(f"Invalid degree '{degree_str}' in '{chord}'. Use I, II, III, IV, V, VI, or VII.")
            degree = degree_map[degree_str]
            result.append((note, quality, degree, int(edo), duration))
        else:
            print(f"Invalid chord format: {chord}")
    return result

def note_to_steps(note, reference="A4"):
    """Convert note (e.g., 'A#') to steps from A4 in 12-EDO."""
    note_map = {'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4, 'F#': -3, 
                'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2}
    return note_map[note]

def get_chord_steps(quality, voices):
    if quality == "major":
        return [0, 4, 7][:voices]
    elif quality == "m":
        return [0, 3, 7][:voices]
    elif quality == "sus":
        return [0, 5, 7][:voices]
    elif quality == "7":
        return [0, 4, 7, 10][:voices]
    else:
        raise ValueError(f"Unknown chord quality: {quality}")

def generate_chord_progression(chord_str, bpm, sample_file="sample.wav", voices=3):
    sr, sample = load_audio(sample_file, mono=True)
    if sr != SAMPLE_RATE:
        sample = resample_audio(sample, sr, SAMPLE_RATE)
    
    chords = parse_chord_string(chord_str)
    print(f"Parsed chords: {chords}")
    
    seconds_per_beat = 60 / bpm
    total_duration = sum(dur * seconds_per_beat for _, _, _, _, dur in chords)
    total_samples = int(total_duration * SAMPLE_RATE)
    output = np.zeros(total_samples, dtype=np.float32)
    print(f"Total duration: {total_duration:.3f}s, total samples: {total_samples}")
    
    current_sample = 0
    for note, quality, degree, n_edo, duration in chords:
        dur_sec = duration * seconds_per_beat
        dur_samples = int(dur_sec * SAMPLE_RATE)
        chord_name = f"{note}{quality if quality != 'major' else ''}-{['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'][degree]}-{n_edo}-{duration}"
        print(f"Chord {chord_name}: {dur_sec:.3f}s, {dur_samples} samples")
        
        steps_12edo = note_to_steps(note)
        base_freq = 440 * (2 ** (steps_12edo / 12))
        steps_from_a4 = n_edo * np.log2(base_freq / 440)
        
        root_steps = steps_from_a4 + degree
        chord_steps = get_chord_steps(quality, voices)
        chord_audio = np.zeros(dur_samples, dtype=np.float32)
        
        for step in chord_steps:
            total_steps = root_steps + (step * n_edo / 12)
            target_freq = 440 * (2 ** (total_steps / n_edo))
            print(f"  Pitching {note} (degree {degree}, step {step}) to {target_freq:.2f} Hz in {n_edo}-EDO")
            pitched = pitch_shift(sample, SAMPLE_RATE, target_freq)
            if len(pitched) > dur_samples:
                pitched = pitched[:dur_samples]
            elif len(pitched) < dur_samples:
                pitched = np.pad(pitched, (0, dur_samples - len(pitched)), 'constant')
            chord_audio += pitched / len(chord_steps)
        
        end_sample = min(current_sample + dur_samples, total_samples)
        print(f"  Placing at {current_sample} to {end_sample}")
        output[current_sample:end_sample] += chord_audio[:end_sample - current_sample]
        current_sample += dur_samples
    
    return output

def main(chord_str, bpm, output_file="progression_with_degrees.wav"):
    print(f"Generating progression at {bpm} BPM...")
    audio = generate_chord_progression(chord_str, bpm)
    save_wav(output_file, audio)

if __name__ == "__main__":
    chord_str = (
    # A1 (Bars 1-8): Moody opening with new EDOs
    "Cm-I-31-1/4, G-IV-19-1/4, A#-VII-12-1/4, F7-II-24-1/4, "    # 1-4
    "Cm-III-43-1/4, Gsus-I-22-1/4, D-V-53-1/4, F-IV-19-1/4, "     # 5-8
    
    # A2 (Bars 9-16): Variation, building tension
    "Cm-I-31-1/4, G7-VI-22-1/4, A#-III-12-1/4, F-II-43-1/4, "     # 9-12
    "Cm-V-53-1/4, D-II-24-1/4, Gsus-IV-19-1/4, F7-VII-31-1/4, "   # 13-16
    
    # B (Bars 17-24): Bright, exotic contrast
    "E-I-22-1/4, B7-IV-43-1/4, F#-VI-12-1/4, C-III-53-1/4, "      # 17-20
    "E-V-19-1/4, Bsus-II-31-1/4, G-VII-24-1/4, D7-I-22-1/4, "     # 21-24
    
    # A3 (Bars 25-32): Return and resolve
    "Cm-I-31-1/4, G-III-19-1/4, A#-V-12-1/4, F-IV-43-1/4, "       # 25-28
    "Cm-VI-53-1/4, Gsus-I-22-1/4, D-IV-24-1/4, F7-II-31-1/4"      # 29-32
)   
    bpm = 10
    main(chord_str, bpm)