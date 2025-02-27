import numpy as np
import pytest
import soundfile as sf
from audio_dsp.synth.subtractive_synth import SubtractiveSynth

@pytest.fixture
def synth():
    """Create and configure a test instance of SubtractiveSynth."""
    s = SubtractiveSynth()
    s.osc_wave = "saw"
    s.filter_cutoff = 800
    s.lfo_freq = 3
    s.lfo_depth = 0.3
    s.lfo_target = "filter"
    return s

def test_waveform_saving(synth, tmp_path):
    """Test if the waveform can be correctly saved as a WAV file."""
    wave = synth.synthesize(freq=220, duration=2.0, attack=0.02, decay=0.1, sustain=0.6, release=0.3)

    # Save as float32 to avoid dtype mismatch
    wav_path = tmp_path / "test_synth_output.wav"
    sf.write(str(wav_path), wave.astype(np.float32), samplerate=synth.sample_rate)

    # Reload and verify the saved file
    loaded_wave, sr = sf.read(str(wav_path))

    assert sr == synth.sample_rate, "Saved WAV file should have the correct sample rate"

    # Convert both to the same dtype and use a reasonable tolerance
    loaded_wave = loaded_wave.astype(wave.dtype)
    assert np.allclose(loaded_wave, wave, atol=1e-4), "Saved waveform should match original"
