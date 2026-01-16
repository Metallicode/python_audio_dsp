import numpy as np
from audio_dsp.utils import wav_io as wavfile
import librosa
from pydub import AudioSegment
import os

def lofi_effect(input_signal, sample_rate=44100, drive=5.0, quantize_q=0.3, 
                reduced_sample_rate=8000, kbps_rate=64, mp3_iterations=1):
    """
    Apply a lo-fi effect chain to an input signal.
    
    Args:
        input_signal: Input audio array (mono, normalized to ±1)
        sample_rate: Input/output sample rate in Hz (default 44100)
        drive: Overdrive gain factor (e.g., 5.0 = strong distortion)
        quantize_q: Quantization step size (e.g., 0.3 = gritty reduction)
        reduced_sample_rate: Temporary sample rate in Hz (e.g., 8000 = lo-fi crunch)
        kbps_rate: Simulated bitrate in kbps (e.g., 64 = muffled loss)
        mp3_iterations: Number of MP3 encode/decode cycles (e.g., 1–100)
    
    Returns:
        Output audio array with lo-fi effects applied
    """
    # Ensure input is float64 and normalized
    signal = np.array(input_signal, dtype=np.float64)
    signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else signal
    total_samples = len(signal)
    
    # Low-pass filter for anti-aliasing (helper function)
    def apply_antialias(signal, cutoff):
        freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
        fft = np.fft.fft(signal)
        fft[np.abs(freqs) > cutoff] = 0  # Sharp cutoff
        return np.real(np.fft.ifft(fft))
    
    # Overdrive
    signal = [1 - np.exp(-x * drive) if x > 0 else -1 + np.exp(x * drive) for x in signal]
    signal = np.array(signal, dtype=np.float64)
    signal = signal / np.max(np.abs(signal))
    print(f"After Overdrive max: {np.max(np.abs(signal)):.5f}")
    
    # Quantization
    signal = quantize_q * np.round(signal / quantize_q)
    signal = signal / np.max(np.abs(signal))
    print(f"After Quantization max: {np.max(np.abs(signal)):.5f}")
    
    # Sample Rate Reducer
    # Anti-alias filter before downsampling
    signal = apply_antialias(signal, reduced_sample_rate / 2)  # Nyquist for target rate
    reduced_samples = int(total_samples * reduced_sample_rate / sample_rate)
    reduced_signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=reduced_sample_rate)
    signal = librosa.resample(reduced_signal, orig_sr=reduced_sample_rate, target_sr=sample_rate)
    signal = signal[:total_samples]  # Trim/pad
    signal = signal / np.max(np.abs(signal))
    print(f"After Sample Rate Reducer max: {np.max(np.abs(signal)):.5f}")
    
    # KBPS Reducer
    kbps_sample_rate = min(11025, int(kbps_rate * 1000 / 32))  # Rough kbps to Hz
    signal = apply_antialias(signal, kbps_sample_rate / 2)  # Anti-alias
    kbps_signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=kbps_sample_rate)
    signal = librosa.resample(kbps_signal, orig_sr=kbps_sample_rate, target_sr=sample_rate)
    signal = signal[:total_samples]
    signal = signal / np.max(np.abs(signal))
    print(f"After KBPS Reducer max: {np.max(np.abs(signal)):.5f}")
    
    # MP3er
    if mp3_iterations > 0:
        wavfile.write("temp.wav", sample_rate, signal.astype(np.float32))
        audio = AudioSegment.from_wav("temp.wav")
        for _ in range(mp3_iterations):
            audio.export("temp.mp3", format="mp3", bitrate=f"{kbps_rate}k")
            audio = AudioSegment.from_mp3("temp.mp3")
        samples = audio.get_array_of_samples()
        signal = np.array(samples, dtype=np.float64) / 32768.0
        signal = librosa.resample(signal, orig_sr=audio.frame_rate, target_sr=sample_rate)
        signal = signal[:total_samples]
        signal = signal / np.max(np.abs(signal))
        print(f"After MP3er max: {np.max(np.abs(signal)):.5f}")
    
    # Clean up temp files
    if os.path.exists("temp.wav"):
        os.remove("temp.wav")
    if os.path.exists("temp.mp3"):
        os.remove("temp.mp3")
    
    return signal

# Test it
if __name__ == "__main__":
    # Load sample data
    samplerate, data = wavfile.read("sequence.wav")
    if samplerate != 44100:
        data = librosa.resample(data.astype(np.float64), orig_sr=samplerate, target_sr=44100)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data / np.max(np.abs(data))  # Normalize
    
    # Apply lo-fi effect
    lofi = lofi_effect(data, sample_rate=44100, drive=1.0, quantize_q=0.3, 
                       reduced_sample_rate=30000, kbps_rate=156, mp3_iterations=5)
    
    # Save output
    wavfile.write("lofi_effect.wav", 44100, lofi.astype(np.float32))
    print(f"Lo-fi effect audio saved to lofi_effect.wav")
