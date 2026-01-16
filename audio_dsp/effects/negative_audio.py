import numpy as np
from audio_dsp.utils import wav_io as wavfile
from scipy.signal import butter, sosfiltfilt
import os

def create_negative_waveform(input_signal):
    """Create a negative waveform by mapping x(t) to 1 - x(t)."""
    max_abs = np.max(np.abs(input_signal))
    if max_abs == 0:
        return input_signal
    signal = input_signal / max_abs
    output = 1 - signal
    output = np.clip(output, -1, 1)
    output = output * max_abs
    return output

def smooth_signal(signal, fs, cutoff_hz=20):
    """Apply low-pass filter to smooth a signal."""
    nyquist = fs / 2
    sos = butter(4, cutoff_hz / nyquist, btype='low', output='sos')
    return sosfiltfilt(sos, np.abs(signal))

def sidechain_compressor(input_signal, control_signal, fs, threshold=0.2, ratio=10.0, attack_ms=5, release_ms=50):
    """Apply sidechain compression using control signal."""
    # Use absolute control signal without normalization
    control = np.abs(control_signal)
    
    # Amplify control signal for stronger effect
    control = control * 10.0
    
    # Smooth control signal
    control = smooth_signal(control, fs, cutoff_hz=100)
    
    # Initialize gain
    gain = np.ones_like(input_signal)
    
    # Attack and release coefficients
    attack_coeff = np.exp(-1.0 / (attack_ms * fs / 1000))
    release_coeff = np.exp(-1.0 / (release_ms * fs / 1000))
    envelope = 0
    
    # Compute gain reduction
    for i in range(len(control)):
        if control[i] > threshold:
            target_gain = 1 - (control[i] - threshold) / ratio
            target_gain = max(target_gain, 0.01)  # Allow deep reduction
        else:
            target_gain = 1.0
        
        # Smooth gain
        if target_gain < envelope:
            coeff = attack_coeff
        else:
            coeff = release_coeff
        envelope = coeff * envelope + (1 - coeff) * target_gain
        gain[i] = envelope
    
    # Apply gain to input
    output = input_signal * gain
    
    # Normalize output to match input RMS
    input_rms = np.sqrt(np.mean(input_signal**2))
    output_rms = np.sqrt(np.mean(output**2))
    if output_rms > 0:
        output = output * (input_rms / output_rms)
    
    return output, gain, control

def visualize_waveforms(input_signal, negative_signal, compressed_signal, gain_signal, control_signal, fs):
    """Plot input, negative, compressed, gain, and control waveforms."""
    import matplotlib.pyplot as plt  # Lazy import for optional dependency
    time = np.arange(len(input_signal)) / fs

    plt.figure(figsize=(12, 20))
    
    plt.subplot(5, 1, 1)
    plt.plot(time, input_signal, label='Input Waveform')
    plt.title('Input Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(5, 1, 2)
    plt.plot(time, negative_signal, label='Negative Waveform', color='orange')
    plt.title('Negative Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(5, 1, 3)
    plt.plot(time, compressed_signal, label='Compressed Waveform', color='purple')
    plt.title('Compressed Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(5, 1, 4)
    plt.plot(time, gain_signal, label='Gain Reduction', color='green')
    plt.title('Gain Reduction')
    plt.xlabel('Time (s)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(5, 1, 5)
    plt.plot(time, control_signal, label='Control Signal', color='red')
    plt.axhline(y=0.2, color='gray', linestyle='--', label='Threshold')
    plt.title('Control Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('waveform_comparison.png')
    plt.close()

def visualize_spectrum(input_signal, negative_signal, compressed_signal, fs):
    """Plot input, negative, and compressed spectra."""
    import matplotlib.pyplot as plt  # Lazy import for optional dependency
    n = len(input_signal)
    freq = np.fft.rfftfreq(n, 1 / fs)
    input_spec = 20 * np.log10(np.abs(np.fft.rfft(input_signal)) + 1e-10)
    negative_spec = 20 * np.log10(np.abs(np.fft.rfft(negative_signal)) + 1e-10)
    compressed_spec = 20 * np.log10(np.abs(np.fft.rfft(compressed_signal)) + 1e-10)

    plt.figure(figsize=(12, 6))
    plt.plot(freq, input_spec, label='Input Spectrum')
    plt.plot(freq, negative_spec, label='Negative Spectrum', linestyle='--')
    plt.plot(freq, compressed_spec, label='Compressed Spectrum', linestyle=':')
    plt.title('Input vs. Negative vs. Compressed Spectra')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.legend()
    plt.savefig('spectrum_comparison.png')
    plt.close()

def main():
    input_file = "input.wav"
    negative_output_file = "negative_output.wav"
    compressed_output_file = "compressed_output.wav"
    
    # Load input WAV
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        return
    fs, input_signal = wavfile.read(input_file)
    if len(input_signal.shape) > 1:
        input_signal = input_signal[:, 0]
    input_signal = input_signal.astype(float) / 32767.0
    
    # Create negative waveform
    print("Creating negative waveform...")
    negative_signal = create_negative_waveform(input_signal)
    
    # Apply sidechain compression using negative signal as control
    print("Applying sidechain compression...")
    compressed_signal, gain_signal, control_signal = sidechain_compressor(
        input_signal, negative_signal, fs,
        threshold=0.2, ratio=10.0, attack_ms=5, release_ms=50
    )
    
    # Visualize
    visualize_waveforms(input_signal, negative_signal, compressed_signal, gain_signal, control_signal, fs)
    visualize_spectrum(input_signal, negative_signal, compressed_signal, fs)
    
    # Save outputs
    negative_signal = (negative_signal * 32767).astype(np.int16)
    compressed_signal = (compressed_signal * 32767).astype(np.int16)
    wavfile.write(negative_output_file, fs, negative_signal)
    wavfile.write(compressed_output_file, fs, compressed_signal)
    print(f"Created: {negative_output_file}, {compressed_output_file}")
    print("Visualizations saved as 'waveform_comparison.png' and 'spectrum_comparison.png'")

if __name__ == "__main__":
    main()