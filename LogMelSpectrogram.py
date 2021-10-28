#!/usr/bin/env python
# coding: utf-8

#   This software component is licensed by ST under BSD 3-Clause license,
#   the "License"; You may not use this file except in compliance with the
#   License. You may obtain a copy of the License at:
#                        https://opensource.org/licenses/BSD-3-Clause


"""LogMel Feature Extraction example."""

import numpy as np
import sys
import librosa
import librosa.display
import scipy.fftpack as fft

SR = 16000
N_FFT = 1024
N_MELS = 10


def create_col(y):
    assert y.shape == (1024,)

    # Create time-series window
    fft_window = librosa.filters.get_window('hann', N_FFT, fftbins=True)
    assert fft_window.shape == (1024,), fft_window.shape

    # Hann window
    y_windowed = fft_window * y
    assert y_windowed.shape == (1024,), y_windowed.shape

    # FFT
    fft_out = fft.fft(y_windowed, axis=0)[:513]
    assert fft_out.shape == (513,), fft_out.shape

    # Power spectrum
    S_pwr = np.abs(fft_out)**2

    assert S_pwr.shape == (513,)

    # Generation of Mel Filter Banks
    mel_basis = librosa.filters.mel(SR, n_fft=N_FFT, n_mels=N_MELS, htk=False)
    assert mel_basis.shape == (10, 513)

    # Apply Mel Filter Banks
    S_mel = np.dot(mel_basis, S_pwr)
    S_mel.astype(np.float32)
    assert S_mel.shape == (10,)

    return S_mel


def feature_extraction(y):
    assert y.shape == (12, 1024)

    S_mel = np.empty((10, 12), dtype=np.float32, order='C')
    for col_index in range(0, 12):
        S_mel[:, col_index] = create_col(y[col_index])

    # Scale according to reference power
    S_mel = S_mel / S_mel.max()
    # Convert to dB
    S_log_mel = librosa.power_to_db(S_mel, top_db=80.0)
    assert S_log_mel.shape == (10, 12)

    return S_log_mel
