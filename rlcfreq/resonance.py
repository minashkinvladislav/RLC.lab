"""Resonance analysis of transmission data (S21) with RLC mode models.

Workflow for one measured two-port:

1. :func:`find_modes` – peaks of ``|S21|`` (prominence in dB);
2. :func:`initial_guess` – resonance frequency and loaded Q from the −3 dB
   bandwidth, complex peak value;
3. :func:`fit_mode` – least-squares fit of :class:`~rlcfreq.models.ResonatorModel`
   (textbook series RLC or L + CPE) in a window of ±``span_bw`` bandwidths.

The physical cross-check used in the lab: for a stripline of length ``ℓ``
the n-th mode is at ``f_n = n c / (2 ℓ sqrt(ε_eff))``, and a constant-phase
(constant loss tangent) dielectric predicts a mode-independent Q, whereas a
frequency-independent series resistance predicts ``Q ∝ f``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.signal import find_peaks

from . import fitting
from .models import ResonatorModel

__all__ = ["find_modes", "initial_guess", "fit_mode", "ModeFit", "mode_number", "epsilon_eff"]

C0 = 299_792_458.0


def find_modes(f: np.ndarray, s21: np.ndarray, prominence_db: float = 10.0) -> np.ndarray:
    db = 20 * np.log10(np.abs(s21))
    idx, _ = find_peaks(db, prominence=prominence_db)
    return idx


def initial_guess(f: np.ndarray, s21: np.ndarray, k: int):
    """Peak frequency, −3 dB bandwidth and complex peak value around index k."""
    m = np.abs(s21[k])
    lo = hi = k
    while lo > 0 and np.abs(s21[lo - 1]) > m / np.sqrt(2):
        lo -= 1
    while hi < len(f) - 1 and np.abs(s21[hi + 1]) > m / np.sqrt(2):
        hi += 1
    bw = max(f[hi] - f[lo], f[1] - f[0])
    return f[k], bw, s21[k]


@dataclass
class ModeFit:
    kind: str
    f_r: float
    Q: float
    result: fitting.FitResult
    window: slice
    model: ResonatorModel


def fit_mode(f: np.ndarray, s21: np.ndarray, k: int, kind: str = "T", span_bw: float = 3.0,
             method: str = "trf") -> ModeFit:
    """Fit one resonance (index ``k`` of the peak) with a T or J mode model."""
    f0, bw, peak = initial_guess(f, s21, k)
    sel = np.where((f > f0 - span_bw * bw) & (f < f0 + span_bw * bw))[0]
    window = slice(sel[0], sel[-1] + 1)
    model = ResonatorModel(kind, f_r=f0, f_span=bw)
    Q0 = f0 / bw
    shape0 = Q0 if kind == "T" else 1 - 2 / np.pi * np.arctan(1 / Q0)
    theta0 = np.array([f0, shape0, peak.real, peak.imag, 0.0, 0.0])
    res = fitting.fit(model, f[window], s21[window], theta0=theta0, weights="unit", method=method)
    return ModeFit(kind, float(res.theta[0]), model.q_factor(res.theta), res, window, model)


def mode_number(f_r: float, spacing: float) -> int:
    return int(round(f_r / spacing))


def epsilon_eff(f_r: float, n: int, length_m: float) -> float:
    """``ε_eff = (n c / (2 ℓ f_n))²`` (end-fringing neglected, IPC-TM-650 2.5.5.5)."""
    return float((n * C0 / (2 * length_m * f_r)) ** 2)
