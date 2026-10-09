"""Time-domain solvers for the integer-order and fractional-order models.

Three independent routes from the frequency-domain model to time:

1. **Integer-order ODE** (textbook series RLC)::

       L q̈ + R q̇ + q / C = v(t),       i = q̇,

   integrated with ``scipy.integrate.solve_ivp`` (RK45, DOP853, Radau, BDF,
   LSODA) and compared with the closed-form step response.

2. **Grünwald–Letnikov (GL) scheme** for the fractional series RLC::

       L D^β i + R i + (1/Q) D^{−α} i = v(t),

   with zero initial history (then GL = Riemann–Liouville = Caputo).  First
   order accurate, O(N²) memory sum.

3. **Numerical inverse Laplace transform** (Talbot contour, ``mpmath``) of
   ``Z(s)·I(s)`` for any model built from the circuit DSL – integer or
   fractional, any topology.  Used for multi-scale battery responses where a
   uniform GL grid would be prohibitively long.

A lock-in estimator recovers ``Z(iω)`` from a simulated steady-state response;
it closes the loop *frequency → time → frequency* in the tests.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

__all__ = [
    "gl_weights",
    "gl_derivative",
    "gl_series_rlc",
    "ode_series_rlc",
    "analytic_series_rlc_step",
    "talbot",
    "step_voltage",
    "lockin_impedance",
]


def gl_weights(order: float, n: int) -> np.ndarray:
    """Grünwald–Letnikov weights ``w_j = (−1)^j C(order, j)``, j = 0..n−1.

    Recurrence ``w_0 = 1``, ``w_j = w_{j−1} (1 − (order+1)/j)``.  For a
    negative order the same weights define the fractional integral.
    """
    w = np.empty(n)
    w[0] = 1.0
    for j in range(1, n):
        w[j] = w[j - 1] * (1.0 - (order + 1.0) / j)
    return w


def gl_derivative(f: np.ndarray, h: float, order: float) -> np.ndarray:
    """``D^order f`` on a uniform grid with ``f(t ≤ 0) = 0`` (FFT convolution)."""
    n = len(f)
    w = gl_weights(order, n)
    m = 1 << (2 * n - 1).bit_length()
    conv = np.fft.irfft(np.fft.rfft(w, m) * np.fft.rfft(f, m), m)[:n]
    return conv * h ** (-order)


def gl_series_rlc(R: float, L: float, Q: float, v: Callable[[np.ndarray], np.ndarray] | np.ndarray,
                  t: np.ndarray, alpha: float = 1.0, beta: float = 1.0) -> Dict[str, np.ndarray]:
    """Current of the fractional series RLC driven by voltage ``v(t)``.

    Implicit GL discretisation on the uniform grid ``t`` (``t[0] = 0``)::

        L h^{−β} Σ_j a_j i_{n−j} + R i_n + (h^{α}/Q) Σ_j b_j i_{n−j} = v_n,

    ``a = w^{(β)}``, ``b = w^{(−α)}``.  Solving for ``i_n`` gives an explicit
    recursion.  With α = β = 1 it is the backward-Euler / rectangle-rule scheme
    for the textbook ODE (``Q`` plays the role of ``C``).

    Returns a dict with ``t``, ``i``, the element voltages ``vR``, ``vL``,
    ``vC`` and the source ``v``.
    """
    t = np.asarray(t, dtype=float)
    h = float(t[1] - t[0])
    if not np.allclose(np.diff(t), h, rtol=1e-9, atol=0):
        raise ValueError("GL scheme needs a uniform grid")
    vv = v(t) if callable(v) else np.asarray(v, dtype=float)
    n = len(t)
    a = gl_weights(beta, n) * L * h ** (-beta)
    b = gl_weights(-alpha, n) * h ** alpha / Q
    c = a + b                               # combined memory weights
    c0 = c[0] + R
    i = np.zeros(n)
    # i_0 corresponds to t = 0 with the source switched on at t = 0+
    for k in range(1, n):
        hist = np.dot(c[1:k + 1], i[k - 1::-1]) if k > 0 else 0.0
        i[k] = (vv[k] - hist) / c0
    vL = gl_derivative(i, h, beta) * L
    vC = gl_derivative(i, h, -alpha) / Q
    return dict(t=t, i=i, v=vv, vR=R * i, vL=vL, vC=vC)


def ode_series_rlc(R: float, L: float, C: float, v: Callable[[float], float], t: np.ndarray,
                   method: str = "Radau", rtol: float = 1e-8, atol: float = 1e-12):
    """Integrate ``L q̈ + R q̇ + q/C = v(t)`` as a first-order system.

    State ``x = (q, i)``; ``ẋ = (i, (v − R i − q/C)/L)``.  The Jacobian is
    supplied for the implicit methods.
    """
    A = np.array([[0.0, 1.0], [-1.0 / (L * C), -R / L]])

    def rhs(tt, x):
        return A @ x + np.array([0.0, v(tt) / L])

    kw = {"jac": (lambda tt, x: A)} if method in {"Radau", "BDF", "LSODA"} else {}
    return solve_ivp(rhs, (t[0], t[-1]), [0.0, 0.0], t_eval=t, method=method, rtol=rtol, atol=atol, **kw)


def analytic_series_rlc_step(R: float, L: float, C: float, V0: float, t: np.ndarray) -> np.ndarray:
    """Exact current after a voltage step ``V0`` at ``t = 0`` (zero state).

    ``I(s) = V0 / (s Z(s)) = (V0/L) / ((s − s₁)(s − s₂))`` with the roots
    ``s₁,₂ = −γ ± sqrt(γ² − ω₀²)``, ``γ = R/(2L)``, ``ω₀² = 1/(LC)``, hence
    ``i(t) = (V0/L) (e^{s₁t} − e^{s₂t}) / (s₁ − s₂)`` (complex arithmetic covers
    the under- and over-damped cases; the critical case is the limit).
    """
    t = np.asarray(t, dtype=float)
    g = R / (2 * L)
    w0sq = 1.0 / (L * C)
    disc = complex(g * g - w0sq) ** 0.5
    if abs(disc) < 1e-12 * max(g, 1.0):
        return V0 / L * t * np.exp(-g * t)
    s1, s2 = -g + disc, -g - disc
    return np.real(V0 / L * (np.exp(s1 * t) - np.exp(s2 * t)) / (s1 - s2))


def talbot(F: Callable, t: Sequence[float], dps: int = 30) -> np.ndarray:
    """Inverse Laplace transform ``f(t) = L⁻¹{F}(t)`` (Talbot contour).

    ``F`` must accept mpmath complex numbers.  Branch cuts of ``s^α`` on the
    negative real axis are admissible for the Talbot contour.
    """
    with mp.workdps(dps):
        return np.array([float(mp.invertlaplace(F, float(tt), method="talbot")) for tt in t])


def step_voltage(model, theta: Sequence[float], t: Sequence[float], I0: float = 1.0,
                 drop_inductive: bool = True, dps: int = 30) -> np.ndarray:
    """Voltage response ``v(t) = L⁻¹{Z(s) I0 / s}(t)`` to a current step ``I0``.

    Works for any :class:`~rlcfreq.models.ImpedanceModel` (see
    :meth:`~rlcfreq.models.ImpedanceModel.mpmath_function` for the treatment
    of inductive elements).
    """
    Z = model.mpmath_function(theta, drop_inductive=drop_inductive)
    return talbot(lambda s: Z(s) * I0 / s, t, dps=dps)


def lockin_impedance(t: np.ndarray, v: np.ndarray, i: np.ndarray, omega: float,
                     n_periods: int = 5) -> complex:
    """Complex impedance at ``omega`` from the last ``n_periods`` of a record.

    The complex amplitudes ``V̂ = (2/T) ∫ v e^{−iωt} dt`` and ``Î`` are computed
    over an integer number of periods at the end of the record (transient
    assumed to have decayed); ``Z = V̂ / Î``.
    """
    T = 2 * np.pi / omega
    mask = t >= t[-1] - n_periods * T
    tt = t[mask]
    e = np.exp(-1j * omega * tt)
    Vh = np.trapezoid(v[mask] * e, tt)
    Ih = np.trapezoid(i[mask] * e, tt)
    return complex(Vh / Ih)
