import librosa
import numpy as np


def load_audio_as_np(path, target_sr=16000):
    """
    Load audio file, convert to mono, resample to target_sr,
    return float32 1D numpy array.
    """
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio.astype(np.float32), target_sr
