import numpy as np
import wave
import struct
import random

def generate_56k_dialup(filename="56k_dialup_random.wav", 
                       duration=8.0, 
                       sample_rate=44100):
    """
    Generate a .wav file with randomized 56k modem-like handshake.
    All timings and frequencies are randomized within reasonable ranges.
    """
    total_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, total_samples, False)
    signal = np.zeros(total_samples)

    # Helper function to add tone segment
    def add_tone(start_time, dur, freq, amp=0.5):
        start_idx = int(start_time * sample_rate)
        end_idx = int((start_time + dur) * sample_rate)
        if end_idx > total_samples:
            end_idx = total_samples
        segment_t = t[start_idx:end_idx]
        tone = amp * np.sin(2 * np.pi * freq * segment_t)
        signal[start_idx:end_idx] += tone

    # Current position in timeline
    current_time = 0.0

    # Phase 1: Initial dial tone
    dial_duration = random.uniform(0.3, 0.7)
    dial_freq = random.uniform(400, 600)
    add_tone(current_time, dial_duration, dial_freq)
    current_time += dial_duration + random.uniform(0.1, 0.3)

    # Phase 2: Answer tone
    answer_duration = random.uniform(0.6, 3.0)
    answer_freq = random.uniform(2000, 2200)
    add_tone(current_time, answer_duration, answer_freq)
    current_time += answer_duration + random.uniform(0.1, 0.4)

    # Phase 3: V.8 bis tones (random number of dual tones)
    tone_count = random.randint(2, 5)
    for _ in range(tone_count):
        if current_time >= duration:
            break
        low_freq = random.uniform(1300, 1500)
        high_freq = random.uniform(1900, 2100)
        tone_duration = random.uniform(0.1, 0.3)
        add_tone(current_time, tone_duration, low_freq)
        add_tone(current_time, tone_duration, high_freq)
        current_time += tone_duration + random.uniform(0.05, 0.2)

    # Phase 4: Training sequence (random chirps)
    chirp_count = random.randint(6, 12)
    for _ in range(chirp_count):
        if current_time >= duration:
            break
        chirp_duration = random.uniform(0.03, 0.07)
        chirp_freq = random.uniform(800, 1800)
        add_tone(current_time, chirp_duration, chirp_freq, 0.3)
        current_time += chirp_duration + random.uniform(0.02, 0.1)

    # Phase 5: Main handshake (random modulated tones)
    handshake_count = random.randint(10, 20)
    for _ in range(handshake_count):
        if current_time >= duration:
            break
        tone_duration = random.uniform(0.1, 0.25)
        base_freq = random.uniform(1000, 1600)
        mod_freq = random.uniform(40, 60)
        start_idx = int(current_time * sample_rate)
        end_idx = int((current_time + tone_duration) * sample_rate)
        if end_idx > total_samples:
            end_idx = total_samples
        segment_t = t[start_idx:end_idx]
        mod = 0.3 * np.sin(2 * np.pi * mod_freq * segment_t)
        tone = 0.4 * np.sin(2 * np.pi * base_freq * segment_t)
        signal[start_idx:end_idx] += tone * (1 + mod)
        current_time += tone_duration + random.uniform(0.05, 0.15)

    # Phase 6: Final connect tones
    if current_time < duration - 0.5:
        final_duration = random.uniform(0.3, 0.6)
        freq1 = random.uniform(1700, 1900)
        freq2 = random.uniform(2300, 2500)
        add_tone(current_time, final_duration, freq1, 0.3)
        add_tone(current_time, final_duration, freq2, 0.3)

    # Normalize (no noise added)
    signal = signal / (max(abs(signal)) * 1.1)
    signal = (signal * 32767).astype(np.int16)

    # Write WAV
    with wave.open(filename, 'w') as wav_file:
        wav_file.setparams((1, 2, sample_rate, total_samples, 'NONE', 'not compressed'))
        for sample in signal:
            wav_file.writeframes(struct.pack('h', sample))

def main():
    print("Generating randomized 56k dial-up sound...")
    generate_56k_dialup("56k_dialup_random.wav")
    print("Created: 56k_dialup_random.wav")

if __name__ == "__main__":
    main()