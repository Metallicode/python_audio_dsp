# import numpy as np
# from audio_dsp.utils import wav_io as wavfile
# from scipy import signal
# import wave
# import struct

# def spectral_flow_compressor(input_file, output_file, threshold=-20, ratio=4.0, viscosity=0.1, window_size=1024, hop_size=256):
#     """
#     Dynamics compressor using spectral energy flow inspired by fluid dynamics.
    
#     Parameters:
#     - input_file: Input WAV file
#     - output_file: Output WAV file
#     - threshold: dB threshold (default -20)
#     - ratio: Compression ratio (default 4.0)
#     - viscosity: Smoothing factor for energy flow (default 0.1)
#     - window_size: STFT window size (default 1024)
#     - hop_size: STFT hop size (default 256)
#     """
#     # Read WAV
#     sample_rate, data = wavfile.read(input_file)
#     if len(data.shape) > 1:
#         data = np.mean(data, axis=1)
#     data = data.astype(float) / np.iinfo(data.dtype).max

#     # STFT
#     freqs, times, Zxx = signal.stft(data, fs=sample_rate, nperseg=window_size, noverlap=window_size - hop_size)
#     magnitudes = np.abs(Zxx)
#     phases = np.angle(Zxx)
#     energy = magnitudes**2  # Spectral energy

#     # Simplified spectral flow (1D per frequency bin)
#     n_freqs, n_times = energy.shape
#     flow_energy = np.zeros_like(energy)
#     flow_energy[:, 0] = energy[:, 0]  # Initial condition

#     # Numerical approximation of energy flow
#     dt = hop_size / sample_rate  # Time step
#     for t in range(1, n_times):
#         # Gradient of energy (velocity-like term)
#         dE_dt = (energy[:, t] - energy[:, t-1]) / dt
#         # Smoothing term (viscosity)
#         smooth = viscosity * (energy[:, t-1] - 2 * energy[:, t] + (energy[:, t+1] if t+1 < n_times else energy[:, t])) / (dt**2)
#         # Update flow energy
#         flow_energy[:, t] = energy[:, t] + dt * (-dE_dt + smooth)
#     flow_energy = np.clip(flow_energy, 0, None)  # Ensure non-negative

#     # Envelope and gain reduction
#     envelope_db = 20 * np.log10(np.maximum(np.sqrt(flow_energy), 1e-10))
#     gain_db = np.zeros_like(envelope_db)
#     above_threshold = envelope_db > threshold
#     gain_db[above_threshold] = threshold + (envelope_db[above_threshold] - threshold) / ratio - envelope_db[above_threshold]
#     gain_linear = 10 ** (gain_db / 20.0)

#     # Apply gain to STFT magnitudes
#     compressed_magnitudes = magnitudes * gain_linear

#     # Reconstruct signal
#     Zxx_compressed = compressed_magnitudes * np.exp(1j * phases)
#     _, output = signal.istft(Zxx_compressed, fs=sample_rate, nperseg=window_size, noverlap=window_size - hop_size)
#     output = output[:len(data)]  # Trim to original length

#     # Normalize
#     output = output / (np.max(np.abs(output)) * 1.1)
#     output = (output * 32767).astype(np.int16)

#     # Write WAV
#     with wave.open(output_file, 'w') as wav_file:
#         wav_file.setparams((1, 2, sample_rate, len(output), 'NONE', 'not compressed'))
#         for sample in output:
#             wav_file.writeframes(struct.pack('h', sample))

# def main():
#     input_file = "input.wav"  # Replace with your WAV
#     output_file = "spectral_flow_compressed.wav"
#     print("Processing with spectral flow compressor...")
#     spectral_flow_compressor(input_file, output_file, threshold=-20, ratio=4.0, viscosity=0.8)
#     print(f"Created: {output_file}")

# if __name__ == "__main__":
#     main()



import numpy as np
from audio_dsp.utils import wav_io as wavfile
from scipy import signal
import matplotlib.pyplot as plt
import wave
import struct

def spectral_flow_compressor(input_file, output_file, threshold=-20, ratio=4.0, viscosity=0.1, window_size=1024, hop_size=256):
    """
    Dynamics compressor using spectral energy flow with visualization.
    
    Parameters:
    - input_file: Input WAV file
    - output_file: Output WAV file
    - threshold: dB threshold (default -20)
    - ratio: Compression ratio (default 4.0)
    - viscosity: Smoothing factor for energy flow (default 0.1)
    - window_size: STFT window size (default 1024)
    - hop_size: STFT hop size (default 256)
    """
    # Read WAV
    sample_rate, data = wavfile.read(input_file)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    data = data.astype(float) / np.iinfo(data.dtype).max

    # STFT
    freqs, times, Zxx = signal.stft(data, fs=sample_rate, nperseg=window_size, noverlap=window_size - hop_size)
    magnitudes = np.abs(Zxx)
    phases = np.angle(Zxx)
    energy = magnitudes**2  # Spectral energy

    # Simplified spectral flow
    n_freqs, n_times = energy.shape
    flow_energy = np.zeros_like(energy)
    flow_energy[:, 0] = energy[:, 0]  # Initial condition

    # Numerical approximation of energy flow
    dt = hop_size / sample_rate
    for t in range(1, n_times):
        dE_dt = (energy[:, t] - energy[:, t-1]) / dt
        smooth = viscosity * (energy[:, t-1] - 2 * energy[:, t] + (energy[:, t+1] if t+1 < n_times else energy[:, t])) / (dt**2)
        flow_energy[:, t] = energy[:, t] + dt * (-dE_dt + smooth)
    flow_energy = np.clip(flow_energy, 0, None)

    # Envelope and gain reduction
    envelope_db = 20 * np.log10(np.maximum(np.sqrt(flow_energy), 1e-10))
    gain_db = np.zeros_like(envelope_db)
    above_threshold = envelope_db > threshold
    gain_db[above_threshold] = threshold + (envelope_db[above_threshold] - threshold) / ratio - envelope_db[above_threshold]
    gain_linear = 10 ** (gain_db / 20.0)

    # Apply gain
    compressed_magnitudes = magnitudes * gain_linear

    # Reconstruct signal
    Zxx_compressed = compressed_magnitudes * np.exp(1j * phases)
    _, output = signal.istft(Zxx_compressed, fs=sample_rate, nperseg=window_size, noverlap=window_size - hop_size)
    output = output[:len(data)]

    # Normalize and write WAV
    output = output / (np.max(np.abs(output)) * 1.1)
    output = (output * 32767).astype(np.int16)
    with wave.open(output_file, 'w') as wav_file:
        wav_file.setparams((1, 2, sample_rate, len(output), 'NONE', 'not compressed'))
        for sample in output:
            wav_file.writeframes(struct.pack('h', sample))

    # Visualization
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # 1. Original Spectral Energy
    axs[0].pcolormesh(times, freqs, 10 * np.log10(np.maximum(energy, 1e-10)), shading='gouraud', cmap='inferno')
    axs[0].set_title("Original Spectral Energy (dB)")
    axs[0].set_ylabel("Frequency (Hz)")

    # 2. Flow Energy
    axs[1].pcolormesh(times, freqs, 10 * np.log10(np.maximum(flow_energy, 1e-10)), shading='gouraud', cmap='inferno')
    axs[1].set_title("Flow Energy After Spectral Smoothing (dB)")
    axs[1].set_ylabel("Frequency (Hz)")

    # 3. Gain Reduction
    axs[2].pcolormesh(times, freqs, gain_db, shading='gouraud', cmap='viridis')
    axs[2].set_title("Gain Reduction (dB)")
    axs[2].set_ylabel("Frequency (Hz)")

    # 4. Compressed Spectral Energy
    compressed_energy = compressed_magnitudes**2
    axs[3].pcolormesh(times, freqs, 10 * np.log10(np.maximum(compressed_energy, 1e-10)), shading='gouraud', cmap='inferno')
    axs[3].set_title("Compressed Spectral Energy (dB)")
    axs[3].set_ylabel("Frequency (Hz)")
    axs[3].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig("spectral_flow_visualization.png")
    plt.close()

def main():
    input_file = "input.wav"  # Replace with your WAV
    output_file = "spectral_flow_compressed.wav"
    print("Processing with spectral flow compressor...")
    spectral_flow_compressor(input_file, output_file, threshold=-50, ratio=4.0, viscosity=0.1)
    print(f"Created: {output_file}")
    print("Visualization saved as 'spectral_flow_visualization.png'")

if __name__ == "__main__":
    main()