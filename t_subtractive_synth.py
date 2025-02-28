import numpy as np
import soundfile as sf
from audio_dsp.synth.subtractive_synth import SubtractiveSynth

s = SubtractiveSynth()
s.osc_wave = "saw"
s.filter_cutoff = 3000  # High cutoff to prevent muting
s.filter_resonance = 5
s.lfo_freq = 0.04
s.filter_q = 5.0
s.lfo_depth = 0.7
#s.filter_type = "bandpass"
#s.filter_type = "highpass"

s.lfo_target = "filter"
wave = s.synthesize(freq=60, duration=50.0, attack=0.02, decay=0.1, sustain=0.6, release=0.3)
wav_path = "test_synth_output.wav"
sf.write(str(wav_path), wave, samplerate=s.sample_rate, subtype='PCM_16')

# # Debugging: Print statistics
# print("Waveform Min:", np.min(wave))
# print("Waveform Max:", np.max(wave))
# print("Waveform Mean:", np.mean(wave))

# # Play Sound
# sd.play(wave, s.sample_rate)
# sd.wait()

# # Save to File
# wav_path = "test_synth_output.wav"
# sf.write(wav_path, wave.astype(np.float32), samplerate=s.sample_rate)

# print(f"Audio file saved as {wav_path}")
