import numpy as np
from audio_dsp.utils import wav_io as wavfile
import mido
import argparse

SAMPLE_RATE = 44100

def save_wav(file_path, data):
    data = np.clip(data, -1, 1)
    wavfile.write(file_path, SAMPLE_RATE, (data * 32767).astype(np.int16))
    print(f"Saved to {file_path}")

def midi_to_notes(midi_file):
    """Extract note events with accurate timing."""
    mid = mido.MidiFile(midi_file)
    ticks_per_beat = mid.ticks_per_beat
    tempo = 500000  # Default 120 BPM
    
    notes = []
    active_notes = {}
    current_time = 0
    
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            if not msg.is_meta:
                current_time += mido.tick2second(msg.time, ticks_per_beat, tempo)
                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = {'note': msg.note, 'time': current_time, 'velocity': msg.velocity / 127}
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        note = active_notes.pop(msg.note)
                        duration = max(current_time - note['time'], 0.01)  # Min 10ms
                        note['duration'] = duration
                        notes.append(note)
    
    for note in active_notes.values():
        note['duration'] = 0.5  # Default for unclosed notes
        notes.append(note)
    
    notes.sort(key=lambda x: x['time'])
    return notes

def midi_to_freq(midi_note, strength=1.0):
    """Convert MIDI note to frequency with logarithmic tuning."""
    semitone_distance = (midi_note - 69) / 12  # Distance from A4 in octaves
    if semitone_distance >= 0:
        exponent = semitone_distance ** strength
    else:
        # Handle negative distances symmetrically
        exponent = -((-semitone_distance) ** strength)
    return 440.0 * (2.0 ** exponent)

def synthesize_note(note, duration, strength=1.0):
    """Generate audio for a MIDI note with logarithmic tuning."""
    freq = midi_to_freq(note['note'], strength)
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, endpoint=False)
    audio = note['velocity'] * np.sin(2 * np.pi * freq * t) * np.exp(-t * 3)
    return audio

def midi_to_wav(midi_file, output_file, strength=1.0):
    """Convert MIDI file to WAV with logarithmic tuning."""
    print(f"Loading MIDI file: {midi_file}...")
    notes = midi_to_notes(midi_file)
    if not notes:
        raise ValueError("No notes found in MIDI file.")
    
    total_duration = max(note['time'] + note['duration'] for note in notes)
    total_samples = int(SAMPLE_RATE * total_duration)
    output_audio = np.zeros(total_samples, dtype=np.float32)
    
    print(f"Total notes: {len(notes)}, duration: {total_duration:.2f}s, samples: {total_samples}")
    
    for note in notes:
        start_sample = int(note['time'] * SAMPLE_RATE)
        audio = synthesize_note(note, note['duration'], strength)
        end_sample = min(start_sample + len(audio), total_samples)
        print(f"Note: {note['note']}, Time: {note['time']:.2f}s, Duration: {note['duration']:.2f}s, Freq: {midi_to_freq(note['note'], strength):.2f} Hz")
        output_audio[start_sample:end_sample] += audio[:end_sample - start_sample]
    
    max_amplitude = np.max(np.abs(output_audio))
    if max_amplitude > 0:
        output_audio /= max_amplitude
    
    save_wav(output_file, output_audio)

def main():
    parser = argparse.ArgumentParser(description="Convert MIDI to WAV with logarithmic tuning.")
    parser.add_argument("midi_file", help="Path to the input MIDI file")
    parser.add_argument("--output", "-o", default="output.wav", help="Output WAV file name")
    parser.add_argument("--strength", "-s", type=float, default=1.0, help="Strength of tuning curve (e.g., 0.5, 1.0, 2.0)")
    args = parser.parse_args()
    
    midi_to_wav(args.midi_file, args.output, args.strength)

if __name__ == "__main__":
    main()