"""Controlled additive noise at a target signal-to-noise ratio."""
from __future__ import annotations

import numpy as np


def power(x: np.ndarray) -> float:
    """Mean power of a signal."""
    return float(np.mean(np.square(x, dtype=np.float64)))


def add_noise_at_snr(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """White Gaussian noise at the given SNR. Power is measured over the whole clip, silences included."""
    p_signal = power(signal)
    if p_signal == 0:
        raise ValueError("cannot set an SNR on a silent signal")
    noise = rng.standard_normal(signal.shape)
    noise *= np.sqrt(p_signal / (10 ** (snr_db / 10)) / power(noise))
    return (signal + noise).astype(np.float32)


def measured_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """SNR of a noisy signal against its clean version."""
    return 10 * np.log10(power(clean) / power(noisy.astype(np.float64) - clean))
