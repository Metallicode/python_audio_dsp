import numpy as np
import soundfile as sf
from audio_dsp.synth.dx7_fm_synth import DX7FMSynth

freq=110
duration=1.0
algorithm=2
sample_rate=44100
filename="dx.wav"

synth = DX7FMSynth(sample_rate=sample_rate)
synth.algorithm = algorithm
synth.freq_ratios =  [1.0, 2.1, 2.3, 4.5] 
synth.feedback = 3.0
audio = synth.synthesize(freq, duration)
sf.write(filename, audio, sample_rate, subtype='PCM_16')
print(f"Sound saved to {filename}")