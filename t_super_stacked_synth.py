import numpy as np
import soundfile as sf
from audio_dsp.synth.super_stacked_synth import SuperStackedSynth

synth = SuperStackedSynth()

# Basic stack: 100 oscillators, equal mix
synth.synthesize(base_freq=110, duration=4.0, num_oscillators=100, detune_spread=0.02, 
                    output_file="super_synth_100.wav")

# Ridiculous stack: 5000 oscillators, saw-heavy
synth.synthesize(base_freq=220, duration=4.0, num_oscillators=5000, detune_spread=0.05, 
                    wave_mix={"sine": 0.1, "triangle": 0.1, "saw": 0.7, "square": 0.1}, 
                    output_file="super_synth_5000.wav")

# Max stack: 10000 oscillators, square only
synth.synthesize(base_freq=440, duration=4.0, num_oscillators=10000, detune_spread=0.03, 
                    wave_mix={"square": 1.0}, output_file="super_synth_10000.wav")