from mido import Message, MidiFile, MidiTrack

def generate_polyrhythmic_midi(
    scale_intervals=[0, 1, 3, 4, 7, 11, 12], 
    base_note=60, 
    base_tempo=100, 
    diff=0.5, 
    note_duration=0.1,
    beat_events=300,
    output_file="polyrhythm_individual_tempos_fixed.mid"
):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    
    # Convert BPM to microseconds per quarter note
    def bpm_to_microseconds_per_beat(bpm):
        return int(60_000_000 / bpm)
    
    # Calculate tempo for each note
    tempos = [base_tempo + (i * diff) for i in range(len(scale_intervals))]
    note_times = [60 / tempo for tempo in tempos]  # seconds per beat
    
    # Convert note times to MIDI ticks
    ticks_per_beat = midi.ticks_per_beat  # Default PPQN (Pulses Per Quarter Note)
    ticks_per_second = ticks_per_beat * (base_tempo / 60)
    note_ticks = [int(time * ticks_per_second) for time in note_times]
    
    # Calculate note off times (short duration)
    note_off_ticks = int(note_duration * ticks_per_second)
    
    events = []
    
    for i, interval in enumerate(scale_intervals):
        pitch = base_note + interval
        tempo = tempos[i]
        tick_interval = note_ticks[i]
        
        time = 0
        while time < ticks_per_beat * beat_events:  # Generate up to 16 beats worth of events
            events.append((time, 'note_on', pitch))
            events.append((time + note_off_ticks, 'note_off', pitch))
            time += tick_interval
    
    # Sort events by time
    events.sort()
    
    last_time = 0
    for event_time, event_type, pitch in events:
        delta_time = event_time - last_time
        last_time = event_time
        
        if event_type == 'note_on':
            track.append(Message('note_on', note=pitch, velocity=64, time=delta_time))
        elif event_type == 'note_off':
            track.append(Message('note_off', note=pitch, velocity=64, time=delta_time))
    
    midi.save(output_file)
    print(f"MIDI file saved: {output_file}")

# Example Usage
generate_polyrhythmic_midi()