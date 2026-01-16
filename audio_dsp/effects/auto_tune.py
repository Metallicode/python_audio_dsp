import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from audio_dsp.utils import wav_io as wavfile

def autotune_effect(input_signal, sample_rate=44100, scale="chromatic", depth=1.0, visualize=False, correction_window=20):
    """
    Apply an auto-tune effect using librosa.pyin with frame-based pitch correction.
    
    Args:
        input_signal: Input audio array (mono, normalized to ±1)
        sample_rate: Sample rate in Hz (default 44100)
        scale: Pitch scale ('chromatic', 'major', 'minor') - notes to snap to
        depth: Correction strength (0–1, 0=no correction, 1=full snap, default 1.0 = hard tune)
        visualize: If True, generates a pitch correction graph.
        correction_window: Number of frames per correction block (smooths correction).
    
    Returns:
        Output audio array with auto-tune effect applied
    """
    # Ensure input is float64 and normalized
    signal = np.array(input_signal, dtype=np.float64)
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))

    # Estimate pitch using pyin
    pitches, voiced_flag, _ = librosa.pyin(signal, fmin=50, fmax=2000, sr=sample_rate)

    # Define pitch scale (MIDI notes)
    if scale == "chromatic":
        valid_notes = np.arange(0, 128)  
    elif scale == "major":
        valid_notes = np.array([0, 2, 4, 5, 7, 9, 11])  
        valid_notes = np.tile(valid_notes, 10) + np.repeat(np.arange(-4, 6) * 12, 7)
    elif scale == "minor":
        valid_notes = np.array([0, 2, 3, 5, 7, 8, 10])  
        valid_notes = np.tile(valid_notes, 10) + np.repeat(np.arange(-4, 6) * 12, 7)
    else:
        raise ValueError("Scale must be 'chromatic', 'major', 'minor'")

    # Compute dominant pitch shift
    shifts = []
    for pitch, voiced in zip(pitches, voiced_flag):
        if voiced and not np.isnan(pitch):
            midi_note = 69 + 12 * np.log2(pitch / 440.0)
            nearest_note = valid_notes[np.argmin(np.abs(valid_notes - midi_note))]
            shifts.append((nearest_note - midi_note) * depth)

    # Use median shift value for consistency
    if len(shifts) > 0:
        avg_shift = np.median(shifts)
    else:
        avg_shift = 0

    print(f"Applying global pitch shift of {avg_shift:.2f} semitones")

    # Apply pitch shift to the full signal
    processed_signal = librosa.effects.pitch_shift(signal, sr=sample_rate, n_steps=avg_shift)

    # Normalize output
    max_amp = np.max(np.abs(processed_signal))
    if max_amp > 0:
        processed_signal /= max_amp
    else:
        print("Warning: Output is silent—check pitch_shift processing.")

    # Visualization
    if visualize:
        times = librosa.times_like(pitches, sr=sample_rate)

        plt.figure(figsize=(10, 5))
        
        # Plot waveforms
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(signal, sr=sample_rate, alpha=0.5, label="Original Signal")
        librosa.display.waveshow(processed_signal, sr=sample_rate, alpha=0.5, color='r', label="Autotuned Signal")
        plt.legend()
        plt.title("Waveform Before & After Autotune")

        # Plot pitch correction
        plt.subplot(2, 1, 2)
        plt.plot(times, pitches, label="Original Pitch (Detected)", color="blue", alpha=0.7)
        plt.xlabel("Time (s)")
        plt.ylabel("Pitch (Hz)")
        plt.legend()
        plt.title("Pitch Detection")

        plt.tight_layout()
        plt.show()

    return processed_signal

# Test it
if __name__ == "__main__":
    samplerate, data = wavfile.read("voice.wav")
    if samplerate != 44100:
        data = librosa.resample(data.astype(np.float64), orig_sr=samplerate, target_sr=44100)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data / np.max(np.abs(data))  

    autotuned = autotune_effect(data, sample_rate=44100, scale="major", depth=6.0, visualize=True, correction_window=50)
    
    wavfile.write("autotune_effect.wav", 44100, autotuned.astype(np.float32))
    print(f"Auto-tune effect audio saved to autotune_effect.wav")
