import numpy as np
import wave
import struct

# ===============================
# PART 1: Scale Generation & Labelling
# ===============================

# Define ideal intervals in cents (for classical intervals)
ideal_intervals = {
    "unison": 0.00,
    "minor second": 111.73,   # roughly 16:15
    "major second": 203.91,   # roughly 9:8
    "minor third": 315.64,    # roughly 6:5
    "major third": 386.31,    # roughly 5:4
    "perfect fourth": 498.04, # roughly 4:3
    "tritone": 611.73,        # one common approximation
    "perfect fifth": 701.96,  # roughly 3:2
    "minor sixth": 813.69,    # roughly 8:5
    "major sixth": 884.36,    # roughly 5:3
    "minor seventh": 1017.60, # roughly 9:5
    "major seventh": 1088.27, # roughly 15:8
    "octave": 1200.00
}

def categorise_interval(cents, threshold=20):
    """
    Compare a given cents value with ideal intervals.
    Returns a tuple of (label, error) if the best match is within threshold,
    otherwise ("Novel interval", error).
    """
    best_label = None
    best_diff = float('inf')
    for label, ideal in ideal_intervals.items():
        diff = abs(cents - ideal)
        if diff < best_diff:
            best_diff = diff
            best_label = label
    if best_diff <= threshold:
        return best_label, best_diff
    else:
        return "Novel interval", best_diff

def generate_scale(root_freq, divisions):
    """
    Generate an equally tempered scale by dividing the octave into 'divisions' steps.
    Returns two lists:
      - frequencies (including the octave),
      - corresponding cent values (0 to 1200).
    """
    step_ratio = 2 ** (1 / divisions)
    frequencies = [root_freq * (step_ratio ** i) for i in range(divisions + 1)]
    cents_values = [1200 * i / divisions for i in range(divisions + 1)]
    return frequencies, cents_values


# ===============================
# PART 2: Synthesis Helpers
# ===============================

def sine_wave(freq, duration, sample_rate=44100, amplitude=0.5):
    """
    Generate a sine wave for a given frequency and duration.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)

def apply_envelope(signal, sample_rate, fade_duration=0.05):
    """
    Apply a simple linear fade-in and fade-out to avoid clicks.
    fade_duration is in seconds.
    """
    fade_samples = int(fade_duration * sample_rate)
    envelope = np.ones_like(signal)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    return signal * envelope


# Helper function: weighted random choice among indices
def weighted_choice(indices, norm_w):
    return np.random.choice(indices, p=norm_w)


if __name__ == "__main__":
    # ===============================
    # Demo: Generate a Random Chord & Melody Progression
    # ===============================

    # Parameters for scale generation
    root_frequency = 440.0  # Hz (A4)
    divisions = 20         # e.g. a 13-tone scale
    tolerance = 20         # cents tolerance when matching ideal intervals

    # Generate the scale and label each note
    freqs, cents_vals = generate_scale(root_frequency, divisions)
    scale = []  # each element will be a dict: frequency, cents, label, error
    for freq, cents in zip(freqs, cents_vals):
        label, error = categorise_interval(cents, threshold=tolerance)
        scale.append({'frequency': freq, 'cents': cents, 'label': label, 'error': error})

    # Print out the scale
    print("Scale notes (from the root):")
    for i, note in enumerate(scale):
        print(f"Note {i+1}: {note['frequency']:.2f} Hz, {note['cents']:.2f} cents, label: {note['label']} (err: {note['error']:.2f})")

    # Define weights for scale degrees based on their consonance.
    consonant_labels = {"unison", "octave", "perfect fifth", "perfect fourth", "major third", "minor third"}
    weights = []
    for note in scale:
        if note['label'] in consonant_labels:
            weights.append(5)
        else:
            weights.append(1)
    weights = np.array(weights, dtype=float)
    norm_weights = weights / weights.sum()  # normalised weights

    num_chords = 30     # number of chord segments
    chord_duration = 5.0  # seconds per chord (and melody note)
    sample_rate = 44100

    chord_progression = []  # will hold list of chords (each chord is a list of note indices)
    melody_notes = []       # one melody note per chord

    all_indices = np.arange(len(scale))

    for _ in range(num_chords):
        # Choose a chord root by weighted random choice
        root_idx = weighted_choice(all_indices, norm_weights)
        # For a simple triad, choose 2 other distinct indices.
        remaining = np.array([i for i in all_indices if i != root_idx])
        rem_weights = weights[remaining]
        rem_norm = rem_weights / rem_weights.sum()
        other_idxs = np.random.choice(remaining, size=2, replace=False, p=rem_norm)
        chord_idxs = sorted([root_idx] + list(other_idxs))
        chord_progression.append(chord_idxs)

        # For the melody note:
        if np.random.rand() < 0.7:
            melody_idx = np.random.choice(chord_idxs)
        else:
            melody_idx = weighted_choice(all_indices, norm_weights)
        melody_notes.append(melody_idx)

    # Print the generated chord progression and melody (with labels)
    print("\nChord Progression (each chord shows indices, frequencies and labels):")
    for i, chord in enumerate(chord_progression):
        chord_info = [f"{scale[idx]['frequency']:.1f}Hz ({scale[idx]['label']})" for idx in chord]
        mel_info = f"{scale[melody_notes[i]]['frequency']:.1f}Hz ({scale[melody_notes[i]]['label']})"
        print(f"Chord {i+1}: " + ", ".join(chord_info) + f" | Melody: {mel_info}")

    # ===============================
    # Synthesis of Audio Example
    # ===============================

    chord_audio_segments = []
    melody_audio_segments = []

    for chord_idxs, mel_idx in zip(chord_progression, melody_notes):
        chord_segment = np.zeros(int(sample_rate * chord_duration))
        for idx in chord_idxs:
            freq = scale[idx]['frequency']
            tone = sine_wave(freq, chord_duration, sample_rate, amplitude=0.3)
            tone = apply_envelope(tone, sample_rate)
            chord_segment += tone
        chord_segment /= len(chord_idxs)
        chord_audio_segments.append(chord_segment)

        mel_freq = scale[mel_idx]['frequency']
        mel_segment = sine_wave(mel_freq, chord_duration, sample_rate, amplitude=0.4)
        mel_segment = apply_envelope(mel_segment, sample_rate)
        melody_audio_segments.append(mel_segment)

    chord_audio = np.concatenate(chord_audio_segments)
    melody_audio = np.concatenate(melody_audio_segments)

    total_samples = len(chord_audio)
    stereo_signal = np.vstack((chord_audio, melody_audio)).T

    max_val = np.abs(stereo_signal).max()
    if max_val > 1:
        stereo_signal /= max_val

    # Write Out to WAV File
    output_filename = "algorithmic_scale_example.wav"
    n_channels = 2
    sampwidth = 2  # bytes (16-bit PCM)
    n_frames = total_samples

    with wave.open(output_filename, 'w') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        for s in stereo_signal:
            left_sample = int(np.clip(s[0], -1, 1) * 32767)
            right_sample = int(np.clip(s[1], -1, 1) * 32767)
            data = struct.pack('<hh', left_sample, right_sample)
            wf.writeframesraw(data)
        wf.writeframes(b'')

    print(f"\nAudio file written to '{output_filename}'. You can now play it to hear the example.")
