import numpy as np
from audio_dsp.utils import wav_io as wavfile
import librosa
import matplotlib.pyplot as plt
import librosa.display

def variable_quantizer_effect(input_signal, sample_rate=44100, frame_size=2048, hop_size=512, min_quant_bits=4, max_quant_bits=16, visualize=False):
    """
    Apply a Variable Quantizer Effect—quantization depth varies with RMS amplitude.
    
    Args:
        input_signal: Input audio array (mono, normalized to ±1)
        sample_rate: Sample rate in Hz (default 44100)
        frame_size: STFT frame size in samples (default 2048)
        hop_size: Hop size in samples (default 512)
        min_quant_bits: Minimum quantization bits (default 4 = 16 levels)
        max_quant_bits: Maximum quantization bits (default 16 = 65536 levels)
        visualize: If True, plot waveform and spectrogram (default False)
    
    Returns:
        Output audio array with variable quantization applied
    """
    # Ensure input is float64 and normalized
    signal = np.array(input_signal, dtype=np.float64)
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
    total_samples = len(signal)
    
    # Frame parameters
    num_frames = (total_samples - frame_size + hop_size) // hop_size + 1
    padded_samples = (num_frames - 1) * hop_size + frame_size
    signal_padded = np.pad(signal, (0, padded_samples - total_samples), mode='constant')
    output = np.zeros_like(signal_padded)
    window_sum = np.zeros_like(signal_padded)
    
    # Process frames
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frame = signal_padded[start:end] * np.hanning(frame_size)
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)), mode='constant')
        
        # Calculate RMS for this frame
        rms = np.sqrt(np.mean(frame**2))  # RMS—signal amplitude
        rms = min(rms, 1.0)  # Cap at 1 (normalized input)
        
        # Map RMS to quantization bits—quiet = coarse, loud = fine
        quant_bits = int(min_quant_bits + (max_quant_bits - min_quant_bits) * rms)  # Linear mapping
        quant_levels = 2 ** quant_bits  # Number of quantization levels
        
        # Quantize frame
        frame_quantized = np.round(frame * quant_levels) / quant_levels
        frame_max = np.max(np.abs(frame_quantized))
        if frame_max > 0:
            frame_quantized /= frame_max  # Pre-normalize frame
        output[start:end] += frame_quantized
        window_sum[start:end] += np.hanning(frame_size)
        
        print(f"Frame {i}: Start {start}-{end}, RMS: {rms:.3f}, Quant bits: {quant_bits}, Quant levels: {quant_levels}, Frame max: {frame_max:.3f}")
    
    # Normalize overlap regions
    output = np.where(window_sum > 0, output / window_sum, output)
    output = output[:total_samples]  # Trim to original length
    
    # Normalize final output
    max_amp = np.max(np.abs(output))
    if max_amp > 0:
        output = output / max_amp
    else:
        print("Warning: Output is silent—check quantization or input signal.")
    print(f"Output range: {np.min(output):.3f} to {np.max(output):.3f}, Output length: {total_samples / sample_rate:.2f}s")
    
    # Visualization
    if visualize:
        times = np.linspace(0, total_samples / sample_rate, total_samples)
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(times, signal, label="Original Signal", alpha=0.5)
        plt.plot(times, output, label="Quantized Signal", color='r', alpha=0.5)
        plt.legend()
        plt.title("Waveform Before & After Variable Quantizer Effect (RMS)")
        plt.subplot(2, 1, 2)
        plt.specgram(output, Fs=sample_rate, NFFT=2048, noverlap=512)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Spectrogram of Quantized Signal")
        plt.tight_layout()
        plt.show()
    
    return output

# Test it
if __name__ == "__main__":
    # Load sample data
    samplerate, data = wavfile.read("input.wav")
    if samplerate != 44100:
        data = librosa.resample(data.astype(np.float64), orig_sr=samplerate, target_sr=44100)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data / np.max(np.abs(data))  # Normalize
    
    # Apply Variable Quantizer Effect with visualization
    effected = variable_quantizer_effect(data, sample_rate=44100, frame_size=1024, hop_size=512, 
                                         min_quant_bits=10, max_quant_bits=16, visualize=True)
    
    # Save output
    wavfile.write("variable_quantizer.wav", 44100, effected.astype(np.float32))
    print(f"Variable Quantizer Effect audio saved to variable_quantizer.wav")