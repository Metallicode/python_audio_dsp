import numpy as np
import scipy.io.wavfile as wav
import os
import random
import librosa
import librosa.effects

def generate_arpeggio(base_freq=100, 
                      multipliers=None, 
                      mode='linear', 
                      octaves=3, 
                      step_size=1, 
                      octave_variation=0.02, 
                      note_length=0.2, 
                      envelope='linear', 
                      arp_tempo=120, 
                      arp_style='up', 
                      output_file='arpeggio.wav', 
                      sample_rate=44100):
    """Generate an arpeggio using custom frequency scaling and styles with octave-dependent variation."""
    
    if multipliers is None:
        multipliers = np.arange(1.5, 5.5, step_size)
    
    if mode == 'exponential':
        multipliers = np.array([1.5 ** i for i in range(1, len(multipliers) + 1)])
    
    # Generate frequency values for each octave with variations
    freq_list = []
    for octave in range(octaves):
        adjusted_multipliers = [m + (octave * octave_variation) for m in multipliers]
        freq_list.extend([base_freq * m * (2 ** octave) for m in adjusted_multipliers])
    
    # Apply arpeggio style
    if arp_style == 'up':
        pass  # Already in ascending order
    elif arp_style == 'down':
        freq_list.reverse()
    elif arp_style == 'altering':
        freq_list = [freq_list[i] if i % 2 == 0 else freq_list[-(i//2+1)] for i in range(len(freq_list))]
    elif arp_style == '2_up_1_back':
        temp_list = []
        for i in range(len(freq_list)):
            if i + 2 < len(freq_list):
                temp_list.extend([freq_list[i], freq_list[i+1], freq_list[i]])
        freq_list = temp_list
    elif arp_style == 'fibonacci':
        fib_seq = [0, 1]
        while len(fib_seq) < len(freq_list):
            fib_seq.append(fib_seq[-1] + fib_seq[-2])
        freq_list = [freq_list[i % len(freq_list)] for i in fib_seq]
    elif arp_style == 'random':
        random.shuffle(freq_list)
    
    # Generate note waveforms
    note_samples = int((60 / arp_tempo) * sample_rate * note_length)
    t = np.linspace(0, note_length, note_samples, endpoint=False)
    
    # Envelope shaping
    if envelope == 'linear':
        env = np.linspace(1, 0, note_samples)
    elif envelope == 'exp':
        env = np.exp(-3 * np.linspace(0, 1, note_samples))
    else:
        env = np.ones(note_samples)  # Flat envelope
    
    arpeggio_wave = np.array([])
    for freq in freq_list:
        wave = np.sin(2 * np.pi * freq * t) * env
        #wave = np.sin(2 * np.pi * ((freq%base_freq)+base_freq) * t) * env #stay in one octave
        arpeggio_wave = np.append(arpeggio_wave, wave)
    
    # Normalize and save file
    arpeggio_wave = np.int16((arpeggio_wave / np.max(np.abs(arpeggio_wave))) * 32767)
    wav.write(output_file, sample_rate, arpeggio_wave)
    print(f"Arpeggio saved: {output_file}")

#[1, 17/16, 6/5, 7/5, 3/2, 8/5, 15/8, 2]

# Example usage
generate_arpeggio(base_freq=150, multipliers=[1, 1.02, 1.04, 1.4,1.7]
, mode='linear', octaves=4, step_size=3, octave_variation=0.5, note_length=0.4, envelope='linear', arp_tempo=320, arp_style='down')