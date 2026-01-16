import numpy as np
from audio_dsp.utils import wav_io as wavfile
import mido
import random

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
                        duration = max(current_time - note['time'], 0.25)
                        note['duration'] = duration
                        notes.append(note)
    
    for note in active_notes.values():
        note['duration'] = 0.5
        notes.append(note)
    
    notes.sort(key=lambda x: x['time'])
    return notes

def synthesize_note(note, duration, speed=1.0, transpose=0):
    """Generate audio for a MIDI note with transposition."""
    freq = 440.0 * (2.0 ** (((note['note'] + transpose) - 69) / 12.0))
    adjusted_duration = duration / speed
    samples = int(SAMPLE_RATE * adjusted_duration)
    t = np.linspace(0, adjusted_duration, samples, endpoint=False)
    audio = note['velocity'] * np.sin(2 * np.pi * freq * t) * np.exp(-t * 3)
    return audio

def midi_looper(midi_file, output_file, start=0, length=16, speed=1.0, reverse=False, 
                loop_count=3, transpose=0, randomise_notes=0.0):
    """Create an audio looper with per-iteration randomisation."""
    print(f"Loading MIDI file: {midi_file}...")
    notes = midi_to_notes(midi_file)
    total_notes = len(notes)
    
    if total_notes == 0:
        raise ValueError("No notes found in MIDI file.")
    
    # Adjust start and length
    start = max(0, min(start, total_notes - 1))
    end = min(start + length, total_notes)
    base_loop_notes = notes[start:end]
    print(f"Total notes in MIDI: {total_notes}, selected {len(base_loop_notes)} from {start} to {end - 1}")
    
    # Calculate loop duration
    loop_start_time = base_loop_notes[0]['time']
    loop_end_time = max(note['time'] + note['duration'] for note in base_loop_notes)
    loop_duration = (loop_end_time - loop_start_time) / speed
    
    print(f"Raw MIDI duration: {loop_start_time:.2f}s to {loop_end_time:.2f}s = {(loop_end_time - loop_start_time):.2f}s")
    print(f"One loop duration (speed-adjusted): {loop_duration:.2f}s")
    loop_samples = int(SAMPLE_RATE * loop_duration)
    print(f"One loop samples: {loop_samples}")
    
    # Prepare full output
    total_samples = loop_samples * loop_count
    output = np.zeros(total_samples, dtype=np.float32)
    
    # Generate each loop iteration
    for i in range(loop_count):
        # Copy base notes for this iteration
        loop_notes = [dict(note) for note in base_loop_notes]  # Deep copy
        
        if reverse:
            loop_notes = loop_notes[::-1]
            for note in loop_notes:
                note['time'] = loop_end_time - (note['time'] - loop_start_time) - note['duration']
        
        if randomise_notes > 0.0:
            note_data = [(note['note'], note['velocity']) for note in loop_notes]
            num_swaps = int(len(note_data) * randomise_notes / 2)
            for _ in range(num_swaps):
                idx1, idx2 = random.sample(range(len(note_data)), 2)
                note_data[idx1], note_data[idx2] = note_data[idx2], note_data[idx1]
            for j, note in enumerate(loop_notes):
                note['note'], note['velocity'] = note_data[j]
        
        print(f"\nLoop iteration {i + 1}/{loop_count} (randomise_notes={randomise_notes}):")
        for note in loop_notes:
            print(f"Note: {note['note']}, Time: {note['time']:.2f}s, Duration: {note['duration']:.2f}s")
        
        # Synthesize this loop
        loop_audio = np.zeros(loop_samples, dtype=np.float32)
        for note in loop_notes:
            start_sample = int((note['time'] - loop_notes[0]['time']) * SAMPLE_RATE / speed)
            audio = synthesize_note(note, note['duration'], speed, transpose)
            end_sample = min(start_sample + len(audio), loop_samples)
            loop_audio[start_sample:end_sample] += audio[:end_sample - start_sample]
        
        # Place in output
        start_idx = i * loop_samples
        output[start_idx:start_idx + loop_samples] = loop_audio
    
    print(f"\nTotal samples: {total_samples}, total duration: {total_samples / SAMPLE_RATE:.2f}s")
    print(f"Generating loop: start={start}, length={length}, speed={speed}, reverse={reverse}, "
          f"transpose={transpose}, randomise_notes={randomise_notes}, total duration={total_samples / SAMPLE_RATE:.2f}s...")
    save_wav(output_file, output)

if __name__ == "__main__":
    midi_file = "midi_files/mir1.mid"
    output_file = "midi_loop.wav"
    start = 50
    length = 10
    speed = 3.0
    reverse = False
    loop_count = 10
    transpose = 0
    randomise_notes = 0.5
    
    midi_looper(midi_file, output_file, start, length, speed, reverse, loop_count, transpose, randomise_notes)

# import numpy as np
# from audio_dsp.utils import wav_io as wavfile
# import mido
# import random

# SAMPLE_RATE = 44100

# def save_wav(file_path, data):
#     data = np.clip(data, -1, 1)
#     wavfile.write(file_path, SAMPLE_RATE, (data * 32767).astype(np.int16))
#     print(f"Saved to {file_path}")

# def midi_to_notes(midi_file):
#     """Extract note events with accurate timing."""
#     mid = mido.MidiFile(midi_file)
#     ticks_per_beat = mid.ticks_per_beat
#     tempo = 500000  # Default 120 BPM
    
#     notes = []
#     active_notes = {}
#     current_time = 0
    
#     for track in mid.tracks:
#         for msg in track:
#             if msg.type == 'set_tempo':
#                 tempo = msg.tempo
#             if not msg.is_meta:
#                 current_time += mido.tick2second(msg.time, ticks_per_beat, tempo)
#                 if msg.type == 'note_on' and msg.velocity > 0:
#                     active_notes[msg.note] = {'note': msg.note, 'time': current_time, 'velocity': msg.velocity / 127}
#                 elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
#                     if msg.note in active_notes:
#                         note = active_notes.pop(msg.note)
#                         duration = max(current_time - note['time'], 0.25)
#                         note['duration'] = duration
#                         notes.append(note)
    
#     for note in active_notes.values():
#         note['duration'] = 0.5
#         notes.append(note)
    
#     notes.sort(key=lambda x: x['time'])
#     return notes

# def synthesize_note(note, duration, speed=1.0, transpose=0):
#     """Generate audio for a MIDI note with transposition."""
#     freq = 440.0 * (2.0 ** (((note['note'] + transpose) - 69) / 12.0))
#     adjusted_duration = duration / speed
#     samples = int(SAMPLE_RATE * adjusted_duration)
#     print(f"Synthesizing note {note['note']} (transposed to {note['note'] + transpose}) at {freq:.2f} Hz, duration {adjusted_duration:.2f}s, {samples} samples")
#     t = np.linspace(0, adjusted_duration, samples, endpoint=False)
#     audio = note['velocity'] * np.sin(2 * np.pi * freq * t) * np.exp(-t * 3)
#     return audio

# def midi_looper(midi_file, output_file, start=0, length=16, speed=1.0, reverse=False, 
#                 loop_count=3, transpose=0, randomise_notes=0.0):
#     """Create an audio looper from a MIDI file with variable randomisation."""
#     print(f"Loading MIDI file: {midi_file}...")
#     notes = midi_to_notes(midi_file)
#     total_notes = len(notes)
    
#     if total_notes == 0:
#         raise ValueError("No notes found in MIDI file.")
    
#     # Adjust start and length
#     start = max(0, min(start, total_notes - 1))
#     end = min(start + length, total_notes)
#     loop_notes = notes[start:end]
#     print(f"Total notes in MIDI: {total_notes}, selected {len(loop_notes)} from {start} to {end - 1}")
    
#     # Calculate loop duration
#     loop_start_time = loop_notes[0]['time']
#     loop_end_time = max(note['time'] + note['duration'] for note in loop_notes)
#     loop_duration = (loop_end_time - loop_start_time) / speed
    
#     if reverse:
#         loop_notes = loop_notes[::-1]
#         for note in loop_notes:
#             note['time'] = loop_end_time - (note['time'] - loop_start_time) - note['duration']
    
#     if randomise_notes > 0.0:
#         # Partial shuffle based on randomise_notes (0.0 to 1.0)
#         note_data = [(note['note'], note['velocity']) for note in loop_notes]
#         num_swaps = int(len(note_data) * randomise_notes / 2)  # Number of pairs to swap
#         for _ in range(num_swaps):
#             i, j = random.sample(range(len(note_data)), 2)  # Pick two random indices
#             note_data[i], note_data[j] = note_data[j], note_data[i]
#         for i, note in enumerate(loop_notes):
#             note['note'], note['velocity'] = note_data[i]
    
#     print(f"Raw MIDI duration: {loop_start_time:.2f}s to {loop_end_time:.2f}s = {(loop_end_time - loop_start_time):.2f}s")
#     print(f"Selected notes (randomise_notes={randomise_notes}):")
#     for note in loop_notes:
#         print(f"Note: {note['note']}, Time: {note['time']:.2f}s, Duration: {note['duration']:.2f}s")
#     print(f"One loop duration (speed-adjusted): {loop_duration:.2f}s")
    
#     # Synthesize one loop
#     loop_samples = int(SAMPLE_RATE * loop_duration)
#     print(f"One loop samples: {loop_samples}")
#     loop_audio = np.zeros(loop_samples, dtype=np.float32)
#     for note in loop_notes:
#         start_sample = int((note['time'] - loop_notes[0]['time']) * SAMPLE_RATE / speed)
#         audio = synthesize_note(note, note['duration'], speed, transpose)
#         end_sample = min(start_sample + len(audio), loop_samples)
#         print(f"Placing note at sample {start_sample} to {end_sample}")
#         loop_audio[start_sample:end_sample] += audio[:end_sample - start_sample]
    
#     # Repeat the loop
#     total_samples = loop_samples * loop_count
#     print(f"Total samples: {total_samples}, total duration: {total_samples / SAMPLE_RATE:.2f}s")
#     output = np.zeros(total_samples, dtype=np.float32)
#     for i in range(loop_count):
#         start_idx = i * loop_samples
#         output[start_idx:start_idx + loop_samples] = loop_audio
    
#     print(f"Generating loop: start={start}, length={length}, speed={speed}, reverse={reverse}, "
#           f"transpose={transpose}, randomise_notes={randomise_notes}, total duration={total_samples / SAMPLE_RATE:.2f}s...")
#     save_wav(output_file, output)

# if __name__ == "__main__":
#     midi_file = "midi_files/can1.mid"
#     output_file = "midi_loop.wav"
#     start = 200
#     length = 15
#     speed = 1.0
#     reverse = False
#     loop_count = 6
#     transpose = 1
#     randomise_notes = 0.1 # 0.0 (none) to 1.0 (full shuffle)
    
#     midi_looper(midi_file, output_file, start, length, speed, reverse, loop_count, transpose, randomise_notes)