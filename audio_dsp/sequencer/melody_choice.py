import numpy as np
from audio_dsp.utils import wav_io as wavfile
import random
import os

# Audio parameters
SAMPLE_RATE = 44100  # Hz
BAR_DURATION = 4.0  # 4 beats per bar (adjustable via BPM)

# Major scale steps (semitones from root)
BASE_FREQ = 261.63  # C4
SCALE_STEPS = [0, 2, 4, 5, 7, 9, 11]  # C major scale
DURATIONS = [0.25, 0.5, 0.75, 1.0]    # Note lengths

def generate_tone(freq, duration):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.sin(freq * t * 2 * np.pi) * 0.5
    fade_samples = int(SAMPLE_RATE * duration * 0.1)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    if fade_samples < len(wave):
        wave[:fade_samples] *= fade_in
        wave[-fade_samples:] *= fade_out
    return wave.astype(np.float32)

def play_audio(wave):
    temp_file = "temp.wav"
    wavfile.write(temp_file, SAMPLE_RATE, (wave * 32767).astype(np.int16))
    if os.name == 'nt':
        import winsound
        winsound.PlaySound(temp_file, winsound.SND_FILENAME)
    else:
        os.system(f"afplay {temp_file}" if os.uname().sysname == 'Darwin' else f"aplay {temp_file}")
    os.remove(temp_file)

def generate_option_tone(current_freq):
    step = random.choice(SCALE_STEPS)
    freq = current_freq * (2 ** (step / 12))
    duration = random.choice(DURATIONS)
    return freq, duration

# Development functions
def horizontal_mirror(melody):
    return melody[::-1]

def vertical_flip(melody, pivot_freq=BASE_FREQ):
    return [(pivot_freq * (pivot_freq / freq), dur) for freq, dur in melody]

def repeat_segment(melody):
    if len(melody) < 2:
        return melody
    start = random.randint(0, len(melody) - 2)
    segment = melody[start:start + 2]
    return melody[:start] + segment + segment + melody[start + 2:]

DEVELOPMENTS = [horizontal_mirror, vertical_flip, repeat_segment]

def develop_melody(base_melody, target_bars=16):
    melody = base_melody[:]
    total_duration = sum(dur for _, dur in melody)
    target_duration = target_bars * BAR_DURATION
    
    while total_duration < target_duration:
        technique = random.choice(DEVELOPMENTS)
        melody = technique(melody)
        total_duration = sum(dur for _, dur in melody)
    
    if total_duration > target_duration:
        excess = total_duration - target_duration
        for i in range(len(melody) - 1, -1, -1):
            if excess > 0:
                dur = melody[i][1]
                if dur <= excess:
                    excess -= dur
                    melody.pop(i)
                else:
                    melody[i] = (melody[i][0], dur - excess)
                    excess = 0
    elif total_duration < target_duration:
        melody.append((BASE_FREQ, target_duration - total_duration))
    
    return melody

def generate_counterpoint(melody, voice_num):
    """Generate a contrapuntal voice, mapping any frequency to the scale."""
    counterpoint = []
    for freq, dur in melody:
        # Calculate semitone distance from BASE_FREQ and map to nearest scale step
        semitone = round(12 * np.log2(freq / BASE_FREQ))
        # Wrap to scale range (0-11 semitones) and adjust by voice
        step_shift = (semitone + random.choice([-2, -1, 1, 2, 3, 4]) * voice_num) % 12
        # Find nearest scale step
        nearest_step = min(SCALE_STEPS, key=lambda x: abs(x - step_shift))
        new_freq = BASE_FREQ * (2 ** (nearest_step / 12))
        counterpoint.append((new_freq, dur))
    return counterpoint

def main():
    # Compose base melody
    melody = [(BASE_FREQ, 1.0)]
    print("Starting melody with C4 (261.63 Hz, 1.0s)")
    play_audio(generate_tone(BASE_FREQ, 1.0))

    while True:
        last_freq = melody[-1][0]
        freq1, dur1 = generate_option_tone(last_freq)
        freq2, dur2 = generate_option_tone(last_freq)

        print(f"\nOption 1: Frequency {freq1:.2f} Hz, Duration {dur1}s")
        play_audio(generate_tone(freq1, dur1))
        print(f"Option 2: Frequency {freq2:.2f} Hz, Duration {dur2}s")
        play_audio(generate_tone(freq2, dur2))

        choice = input("Choose (1 or 2) or 'done': ").strip().lower()
        if choice == 'done':
            break
        elif choice not in ['1', '2']:
            print("Invalid choice.")
            continue

        chosen_freq, chosen_dur = (freq1, dur1) if choice == '1' else (freq2, dur2)
        melody.append((chosen_freq, chosen_dur))
        print(f"Added: {chosen_freq:.2f} Hz, {chosen_dur}s")
        play_audio(np.concatenate([generate_tone(f, d) for f, d in melody]))

    # Develop melody to 16 bars
    bpm = float(input("Enter tempo in BPM (e.g., 120): "))
    global BAR_DURATION
    BAR_DURATION = 60 / bpm * 4
    developed_melody = develop_melody(melody)
    print(f"Developed melody: {[(f'{f:.2f}', d) for f, d in developed_melody]}")

    # Generate 3 voices
    voice1 = developed_melody
    voice2 = generate_counterpoint(voice1, 1)
    voice3 = generate_counterpoint(voice1, -1)

    print(f"Voice 2: {[(f'{f:.2f}', d) for f, d in voice2[:5]]}...")  # First 5 for brevity
    print(f"Voice 3: {[(f'{f:.2f}', d) for f, d in voice3[:5]]}...")

    # Generate audio for each voice
    voice1_wave = np.concatenate([generate_tone(f, d) for f, d in voice1])
    voice2_wave = np.concatenate([generate_tone(f, d) for f, d in voice2])
    voice3_wave = np.concatenate([generate_tone(f, d) for f, d in voice3])

    # Mix voices
    max_len = max(len(voice1_wave), len(voice2_wave), len(voice3_wave))
    mixed_wave = np.zeros(max_len, dtype=np.float32)
    for wave in [voice1_wave, voice2_wave, voice3_wave]:
        mixed_wave[:len(wave)] += wave / 3

    # Save and play
    print("Saving to 'three_voice_composition.wav'")
    wavfile.write("three_voice_composition.wav", SAMPLE_RATE, (mixed_wave * 32767).astype(np.int16))
    print("Playing final composition...")
    play_audio(mixed_wave)
    print("Done!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        