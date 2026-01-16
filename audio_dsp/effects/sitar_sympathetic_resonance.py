import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from audio_dsp.utils import wav_io as wavfile

def sitar_sympathetic_resonance(input_signal, sample_rate=44100, resonant_freqs=None, 
                                decay=1.0, wet_mix=0.5, threshold=12.0, visualize=False):
    """
    Apply a Sitar Sympathetic Resonance effect—multiple resonant frequencies are excited based on input pitch and energy.
    
    Args:
        input_signal: Input audio array (mono, normalized to ±1)
        sample_rate: Sample rate in Hz.
        resonant_freqs: List of sympathetic frequencies in Hz.
        decay: Decay time in seconds for resonant sines.
        wet_mix: Wet signal mix (0–1).
        threshold: MIDI proximity threshold in semitones.
        visualize: If True, plot waveform and spectrogram.
    
    Returns:
        Output audio array with sympathetic resonance.
    """
    # Normalize input
    signal = np.array(input_signal, dtype=np.float64)
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
    total_samples = len(signal)

    # Generate default resonant frequencies if not provided (Broad harmonic series)
    if resonant_freqs is None:
        resonant_freqs = []
        base_freqs = [110, 220, 330, 440, 550, 660, 770, 880]  # Some fundamental tones
        for base in base_freqs:
            for octave in [1, 2, 4, 8]:  # Include octaves
                resonant_freqs.append(base * octave)

    # Reduce hop length for better resolution
    hop_length = 256  
    pitches, voiced_flag, _ = librosa.pyin(signal, fmin=50, fmax=5000, sr=sample_rate, hop_length=hop_length)

    # Compute RMS loudness (match pitch frames)
    rms = librosa.feature.rms(y=signal, hop_length=hop_length)[0]

    # Convert resonant frequencies to MIDI
    resonant_midi = [69 + 12 * np.log2(freq / 440.0) for freq in resonant_freqs]
    print(f"Sympathetic MIDI: {', '.join(f'{midi:.2f}' for midi in resonant_midi)}")

    # Resonance Processing
    wet_output = np.zeros_like(signal)
    decay_samples = int(decay * sample_rate)
    resonance_triggered = False  # Track if any resonance occurs

    for i in range(total_samples):
        # Compute the correct index for `pitches` based on hop length
        pitch_idx = min(i // hop_length, len(pitches) - 1)

        pitch_hz = pitches[pitch_idx] if voiced_flag[pitch_idx] and not np.isnan(pitches[pitch_idx]) else 0
        rms_value = rms[min(pitch_idx, len(rms) - 1)]

        if pitch_hz > 50 and pitch_hz < 5000:
            input_midi = 69 + 12 * np.log2(pitch_hz / 440.0)

            # Generate resonant sines with overlapping frequencies
            t = np.arange(decay_samples) / sample_rate
            resonant_wave = np.zeros(min(decay_samples, total_samples - i))

            for res_midi, res_freq in zip(resonant_midi, resonant_freqs):
                diff = abs(input_midi - res_midi)
                if diff <= threshold:
                    amp = rms_value * (1 - diff / threshold)

                    # **Ensure minimum amplitude so resonance isn't silent**
                    amp = max(amp, 0.1)  # **Boost signal if too weak**

                    harmonics = [res_freq * 2**n for n in range(0, 3)]  # Base + 2nd + 4th harmonic

                    for harmonic in harmonics:
                        sine_wave = amp * np.sin(2 * np.pi * harmonic * t)

                        # **Ensure sizes match before adding**
                        min_len = min(len(resonant_wave), len(sine_wave))
                        resonant_wave[:min_len] += sine_wave[:min_len]

                    resonance_triggered = True  # **Mark resonance as happening**

            # Overlap-add resonance
            start = min(i, total_samples - len(resonant_wave))
            wet_output[start:start+len(resonant_wave)] += resonant_wave

    # **Normalize wet signal separately**
    max_wet_amp = np.max(np.abs(wet_output))
    if max_wet_amp > 0:
        wet_output = wet_output / max_wet_amp  # Normalize to prevent clipping

    # **Proper Wet/Dry Mixing**
    dry_mix = 1.0 - wet_mix
    output = (dry_mix * signal) + (wet_mix * wet_output)

    # Final Normalization
    max_amp = np.max(np.abs(output))
    if max_amp > 0:
        output = output / max_amp
    else:
        print("❌ FINAL WARNING: Output is still silent! Resonance is not triggering properly.")
    
    print(f"Output range: {np.min(output):.3f} to {np.max(output):.3f}, Wet mix: {wet_mix}, Decay: {decay}")

    # Visualization
    if visualize:
        times = np.linspace(0, total_samples / sample_rate, total_samples)
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(times, signal, label="Original Signal (Dry)", alpha=0.5)
        plt.plot(times, wet_output, label="Resonated Signal (Wet)", color='r', alpha=0.5)
        plt.legend()
        plt.title("Waveform Before & After Resonance")
        plt.subplot(2, 1, 2)
        plt.specgram(output, Fs=sample_rate, NFFT=2048, noverlap=512)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Spectrogram of Resonated Signal")
        plt.tight_layout()
        plt.show()
    
    return output

# Test it
if __name__ == "__main__":
    samplerate, data = wavfile.read("input.wav")
    if samplerate != 44100:
        data = librosa.resample(data.astype(np.float64), orig_sr=samplerate, target_sr=44100)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data / np.max(np.abs(data))

    # Define sympathetic frequencies: More high-frequency resonance
    resonant_freqs = [110 * (i / 1.2) for i in range(2, 100)]  # Up to ~15kHz

    # Apply Sitar Sympathetic Resonance
    effected = sitar_sympathetic_resonance(data, sample_rate=44100, resonant_freqs=resonant_freqs, decay=1.0, wet_mix=0.5, threshold=32.0, visualize=True)
    
    # Save output
    wavfile.write("sitar_resonance.wav", 44100, effected.astype(np.float32))
    print(f"Sitar Sympathetic Resonance audio saved to sitar_resonance.wav")
