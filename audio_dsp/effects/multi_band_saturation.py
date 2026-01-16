import numpy as np
from audio_dsp.utils import wav_io as wavfile
from scipy.signal import butter, sosfiltfilt, sosfreqz
import matplotlib.pyplot as plt
import os

def design_filter_bank(crossovers, fs, order=4):
    """Design Butterworth band-pass filters for each frequency band."""
    nyquist = fs / 2
    bands = []
    if crossovers:
        sos = butter(order, crossovers[0] / nyquist, btype='low', output='sos')
        bands.append(('low', 0, crossovers[0], sos))
    for i in range(len(crossovers) - 1):
        low, high = crossovers[i], crossovers[i + 1]
        sos = butter(order, [low / nyquist, high / nyquist], btype='band', output='sos')
        bands.append((f'mid{i+1}', low, high, sos))
    if crossovers:
        sos = butter(order, crossovers[-1] / nyquist, btype='high', output='sos')
        bands.append(('high', crossovers[-1], fs / 2, sos))
    else:
        sos = butter(order, nyquist / nyquist, btype='low', output='sos')
        bands.append(('full', 0, fs / 2, sos))
    return bands

def apply_saturation(signal, drive, fs):
    """Apply tanh-based saturation/overdrive to a signal."""
    max_abs = np.max(np.abs(signal))
    if max_abs == 0 or drive == 0:
        return signal  # Return original signal if silent or drive=0
    signal = signal / max_abs
    drive = np.clip(drive, 0, 10)
    saturated = np.tanh(drive * signal)
    max_saturated = np.max(np.abs(saturated))
    if max_saturated == 0:
        return signal
    return saturated / max_saturated * max_abs

def process_multi_band(input_signal, fs, crossovers, drives):
    """Split signal into bands, apply saturation, and recombine."""
    expected_bands = len(crossovers) + 1
    if len(drives) != expected_bands:
        raise ValueError(f"Expected {expected_bands} drive values for {expected_bands} bands, got {len(drives)}. "
                         f"Provide one drive per band (e.g., add 'high::{0.0}' for the high band).")
    
    filter_bank = design_filter_bank(crossovers, fs)
    output = np.zeros_like(input_signal)
    
    for i, (band_name, low, high, sos) in enumerate(filter_bank):
        band_signal = sosfiltfilt(sos, input_signal)
        drive = drives[i]
        processed = apply_saturation(band_signal, drive, fs)
        output += processed
        print(f"Processed band {band_name} ({low}-{high} Hz) with drive {drive}")
    
    max_abs = np.max(np.abs(output))
    if max_abs > 0:
        output = output / max_abs * 0.9
    return output

def visualize_filters(filter_bank, fs):
    """Plot frequency response of the filter bank using sosfreqz."""
    plt.figure(figsize=(12, 6))
    for band_name, low, high, sos in filter_bank:
        w, h = sosfreqz(sos, worN=2000, fs=fs)
        plt.plot(w, 20 * np.log10(np.abs(h) + 1e-10), label=f'{band_name} ({low}-{high} Hz)')
    plt.title('Filter Bank Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.grid(True)
    plt.legend()
    plt.savefig('filter_bank_response.png')
    plt.close()

def visualize_spectrum(input_signal, output_signal, fs):
    """Plot input vs. output spectrum."""
    n = len(input_signal)
    freq = np.fft.rfftfreq(n, 1 / fs)
    input_spec = 20 * np.log10(np.abs(np.fft.rfft(input_signal)) + 1e-10)
    output_spec = 20 * np.log10(np.abs(np.fft.rfft(output_signal)) + 1e-10)
    
    plt.figure(figsize=(12, 6))
    plt.plot(freq, input_spec, label='Input Spectrum')
    plt.plot(freq, output_spec, label='Output Spectrum')
    plt.title('Input vs. Output Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.legend()
    plt.savefig('spectrum_comparison.png')
    plt.close()

def parse_pattern(pattern):
    """Parse pattern string like 'low:200:2.0,mid:1000:5.0,mid2:5000:3.0,high::0.0'."""
    bands = pattern.split(',')
    crossovers = []
    drives = []
    for band in bands:
        try:
            parts = band.split(':')
            if len(parts) != 3:
                print(f"Invalid band format: {band}, skipping")
                continue
            name, freq, drive = parts
            drive = float(drive)
            if freq:
                freq = float(freq)
                crossovers.append(freq)
            drives.append(drive)
        except ValueError:
            print(f"Invalid band format: {band}, skipping")
    crossovers = sorted(crossovers)
    if len(drives) != len(crossovers) + 1:
        print(f"Warning: Got {len(drives)} drives for {len(crossovers) + 1} bands. Padding with drive=0.0")
        while len(drives) < len(crossovers) + 1:
            drives.append(0.0)
    return crossovers, drives

def main():
    input_file = "sequence.wav"
    output_file = "saturated_output.wav"
    pattern = "low:200:2.0,mid:1000:5.0,mid2:5000:3.0,high::0.0"
    
    crossovers, drives = parse_pattern(pattern)
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        return
    fs, input_signal = wavfile.read(input_file)
    if len(input_signal.shape) > 1:
        input_signal = input_signal[:, 0]
    input_signal = input_signal.astype(float) / 32767.0
    
    print("Processing multi-band saturation...")
    output_signal = process_multi_band(input_signal, fs, crossovers, drives)
    
    filter_bank = design_filter_bank(crossovers, fs)
    visualize_filters(filter_bank, fs)
    visualize_spectrum(input_signal, output_signal, fs)
    
    output_signal = (output_signal * 32767).astype(np.int16)
    wavfile.write(output_file, fs, output_signal)
    print(f"Created: {output_file}")
    print("Visualizations saved as 'filter_bank_response.png' and 'spectrum_comparison.png'")

if __name__ == "__main__":
    main()



'''
The script supports any number of bands. Try:

2 Bands: pattern = "low:200:2.0,high::0.0"
3 Bands: pattern = "low:150:1.0,mid:800:4.0,high::2.0"
5 Bands: pattern = "b1:100:1.0,b2:500:3.0,b3:2000:5.0,b4:8000:2.0,b5::0.0"

'''