import numpy as np
import matplotlib.pyplot as plt
from audio_dsp.utils import wav_io as wavfile
from audio_dsp.utils import load_audio, resample_audio

def reverb_effect(input_signal, ir_path, sample_rate=44100, wet_mix=0.5, pre_delay=0.0, decay_factor=1.0, 
                  hpf_freq=100.0, lpf_freq=10000.0):
    """
    Apply a high-fidelity convolution reverb effect using an impulse response (WAV or AIFF).
    
    Args:
        input_signal: Input audio array (mono, normalized to ±1)
        ir_path: Path to impulse response file (WAV or AIFF)
        sample_rate: Sample rate in Hz (default 44100)
        wet_mix: Wet signal mix (0–1, default 0.5 = 50% reverb)
        pre_delay: Pre-delay in seconds (default 0.0 = no delay)
        decay_factor: Decay time scaling (e.g., 0.5 = shorter, 2.0 = longer, default 1.0)
        hpf_freq: High-pass filter cutoff in Hz (default 100 Hz)
        lpf_freq: Low-pass filter cutoff in Hz (default 10000 Hz)
    
    Returns:
        Output audio array with reverb effect applied
    """
    # Ensure input is float64 and normalized
    signal = np.array(input_signal, dtype=np.float64)
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
    total_samples = len(signal)
    
    # Load impulse response
    ir_rate, ir = load_audio(ir_path, mono=True)
    if ir_rate != sample_rate:
        ir = resample_audio(ir, ir_rate, sample_rate)
    ir = ir / np.max(np.abs(ir))  # Normalize IR
    ir_samples = len(ir)
    print(f"IR loaded: {ir_path}, Samples: {ir_samples}, Rate: {sample_rate}")
    
    # Apply decay factor to IR
    if decay_factor != 1.0:
        decay_env = np.exp(-np.linspace(0, 5 * decay_factor, ir_samples))
        ir *= decay_env
        print(f"Decay factor applied: {decay_factor}")
    
    # Apply pre-delay
    pre_delay_samples = int(pre_delay * sample_rate)
    if pre_delay_samples > 0:
        ir = np.pad(ir, (pre_delay_samples, 0), mode='constant')[:-pre_delay_samples]
        print(f"Pre-delay applied: {pre_delay_samples} samples ({pre_delay:.3f}s)")
    
    # High-pass and low-pass filtering on IR
    if hpf_freq > 0 or lpf_freq < sample_rate / 2:
        freqs = np.fft.fftfreq(ir_samples, 1/sample_rate)
        fft_ir = np.fft.fft(ir)
        filter_mask = np.ones_like(fft_ir)
        if hpf_freq > 0:
            filter_mask *= (np.abs(freqs) >= hpf_freq)  # High-pass
        if lpf_freq < sample_rate / 2:
            filter_mask *= (np.abs(freqs) <= lpf_freq)  # Low-pass
        fft_ir *= filter_mask
        ir = np.real(np.fft.ifft(fft_ir))
        print(f"EQ applied: HPF {hpf_freq} Hz, LPF {lpf_freq} Hz")
    
    # FFT convolution with overlap-add
    frame_size = 65536  # Large blocks for quality—no real-time constraint
    hop_size = frame_size // 2  # 50% overlap for smooth convolution
    num_frames = (total_samples + hop_size - 1) // hop_size + 1
    padded_samples = (num_frames - 1) * hop_size + frame_size + ir_samples - 1
    signal_padded = np.pad(signal, (0, padded_samples - total_samples), mode='constant')
    output = np.zeros_like(signal_padded)
    
    # Pre-compute FFT of IR
    ir_fft = np.fft.fft(ir, n=frame_size + ir_samples - 1)
    
    for i in range(0, total_samples, hop_size):
        start = i
        end = start + frame_size
        if end > padded_samples:
            break
        frame = signal_padded[start:end]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)), mode='constant')
        
        # FFT convolution
        frame_fft = np.fft.fft(frame, n=frame_size + ir_samples - 1)
        conv_fft = frame_fft * ir_fft
        conv_frame = np.real(np.fft.ifft(conv_fft))
        
        # Overlap-add
        output[start:start + len(conv_frame)] += conv_frame
    
    output = output[:total_samples]  # Trim to original length
    
    # Wet/dry mix
    output = signal * (1 - wet_mix) + output * wet_mix
    
    # Normalize
    max_amp = np.max(np.abs(output))
    if max_amp > 0:
        output = output / max_amp
    print(f"Output range: {np.min(output):.3f} to {np.max(output):.3f}, Wet mix: {wet_mix}, Pre-delay: {pre_delay}s, Decay: {decay_factor}")
    
    return output

# Test it
if __name__ == "__main__":
    # Load sample data
    samplerate, data = load_audio("input.wav", mono=True)
    if samplerate != 44100:
        data = resample_audio(data, samplerate, 44100)
        samplerate = 44100
    data = data / np.max(np.abs(data))  # Normalize
    
    ir_dir = "impulse_response/"
    # Apply reverb effect with an impulse response (replace with your AIF IR path)
    ir_path = ir_dir+"3000CStreetGarageStairwell.wav"  # Replace with your AIF IR file path
    reverbed = reverb_effect(data, ir_path, sample_rate=44100, wet_mix=0.5, pre_delay=0.05, decay_factor=1.0, 
                             hpf_freq=100.0, lpf_freq=10000.0)
    
    # Save output as WAV
    wavfile.write("reverb_effect.wav", 44100, reverbed.astype(np.float32))
    print(f"Reverb effect audio saved to reverb_effect.wav")