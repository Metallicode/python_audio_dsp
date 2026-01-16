import numpy as np
from audio_dsp.utils import wav_io as wavfile
import librosa
import matplotlib.pyplot as plt
import librosa.display

def temporal_gravity_warp(input_signal, sample_rate=44100, gravity_factor=2.0, wet_mix=0.5, visualize=False):
    """
    Apply a Temporal Gravity Warp effect—time dilation based on amplitude 'mass'.
    
    Args:
        input_signal: Input audio array (mono, normalized to ±1)
        sample_rate: Sample rate in Hz (default 44100)
        gravity_factor: Strength of time dilation (0–5, default 2.0 = moderate warp)
        wet_mix: Wet signal mix (0–1, default 0.5 = 50% effect)
        visualize: If True, plot waveform and RMS (default False)
    
    Returns:
        Output audio array with temporal gravity warp applied
    """
    # Ensure input is float64 and normalized
    signal = np.array(input_signal, dtype=np.float64)
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
    total_samples = len(signal)
    
    # Pre-pad signal for RMS to avoid length mismatch
    frame_size = 2048  # Frame size for RMS—smooth envelope
    hop_size = frame_size // 4  # ~25% overlap—dense RMS
    rms_num_frames = (total_samples + hop_size - 1) // hop_size + 1
    padded_samples = rms_num_frames * hop_size  # Ensure RMS fits
    signal_padded = np.pad(signal, (0, padded_samples - total_samples), mode='constant')
    
    # Compute RMS envelope
    rms = librosa.feature.rms(y=signal_padded, frame_length=frame_size, hop_length=hop_size)[0]
    rms_frames = len(rms)
    rms_samples = rms_frames * hop_size
    rms = np.interp(np.arange(total_samples), np.linspace(0, total_samples, rms_frames), rms)  # Interpolate to exact length
    rms = np.convolve(rms, np.ones(100) / 100, mode='same')  # Smooth RMS
    
    # Map RMS to speed (time dilation)
    speed = 1.0 / (1.0 + gravity_factor * rms)  # Louder → slower (e.g., 0.5x), Quieter → faster (e.g., 1.5x)
    speed = np.clip(speed, 0.5, 1.5)  # Limit range—avoid extremes
    
    # Compute warped time axis
    time_orig = np.arange(total_samples)
    time_warped = np.cumsum(1.0 / speed)  # Cumulative time stretch
    time_warped = time_warped * (total_samples / time_warped[-1])  # Normalize to original length
    
    # Resample signal to warped time—preserve pitch
    output = np.interp(time_warped, time_orig, signal)  # Linear interpolation—smooth warp
    
    # Wet/dry mix
    output = signal * (1 - wet_mix) + output * wet_mix
    
    # Normalize
    max_amp = np.max(np.abs(output))
    if max_amp > 0:
        output = output / max_amp
    print(f"Output range: {np.min(output):.3f} to {np.max(output):.3f}, Wet mix: {wet_mix}, Gravity factor: {gravity_factor}, "
          f"RMS frames: {rms_frames}, RMS range: {np.min(rms):.3f} to {np.max(rms):.3f}, Speed range: {np.min(speed):.3f} to {np.max(speed):.3f}")
    
    # Visualization
    if visualize:
        times = np.linspace(0, total_samples / sample_rate, total_samples)
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(times, signal, label="Original Signal", alpha=0.5)
        plt.plot(times, output, label="Warped Signal", color='r', alpha=0.5)
        plt.legend()
        plt.title("Waveform Before & After Temporal Gravity Warp")
        plt.subplot(2, 1, 2)
        plt.plot(times, rms, label="RMS (Mass)", color="blue", alpha=0.7)
        plt.plot(times, speed, label="Speed (Dilation)", color="green", alpha=0.7)
        plt.xlabel("Time (s)")
        plt.legend()
        plt.title("RMS and Speed Over Time")
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
    
    # Apply Temporal Gravity Warp with visualization
    effected = temporal_gravity_warp(data, sample_rate=44100, gravity_factor=2.0, wet_mix=0.5, visualize=True)
    
    # Save output
    wavfile.write("gravity_warp.wav", 44100, effected.astype(np.float32))
    print(f"Temporal Gravity Warp audio saved to gravity_warp.wav")