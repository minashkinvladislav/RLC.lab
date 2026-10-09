"""Time-domain solvers against closed forms and against each other."""
import math

import numpy as np
import pytest

from rlcfreq import timedomain as td
from rlcfreq.models import ImpedanceModel, series_rlc

R, L, C, V0 = 10.0, 1e-3, 1e-6, 1.0


def test_gl_weights_integer_orders():
    assert np.allclose(td.gl_weights(1.0, 5), [1, -1, 0, 0, 0])
    assert np.allclose(td.gl_weights(-1.0, 5), [1, 1, 1, 1, 1])
    assert np.allclose(td.gl_weights(2.0, 4), [1, -2, 1, 0])


def test_gl_fractional_integral_of_constant():
    """D^{−α} 1 = t^α / Γ(1+α)."""
    a = 0.6
    t = np.linspace(0, 1, 4001)
    f = np.ones_like(t)
    approx = td.gl_derivative(f, t[1] - t[0], -a)
    exact = t ** a / math.gamma(1 + a)
    late = t >= 0.1                      # the jump of f at t = 0 costs O(h^α) near 0
    assert np.max(np.abs(approx - exact)[late]) < 5e-4
    assert np.max(np.abs(approx[1:] - exact[1:])) < 1e-2


@pytest.mark.parametrize("method", ["RK45", "DOP853", "Radau", "BDF", "LSODA"])
def test_ode_solvers_match_closed_form(method):
    t = np.linspace(0, 2e-3, 801)
    sol = td.ode_series_rlc(R, L, C, lambda tt: V0, t, method=method)
    ex = td.analytic_series_rlc_step(R, L, C, V0, t)
    assert np.max(np.abs(sol.y[1] - ex)) < 1e-7 * max(1.0, np.abs(ex).max() * 1e3)


def test_gl_first_order_convergence():
    errs = []
    for n in (1001, 4001):
        t = np.linspace(0, 2e-3, n)
        g = td.gl_series_rlc(R, L, C, lambda x: np.where(x > 0, V0, 0.0), t)
        errs.append(np.max(np.abs(g["i"] - td.analytic_series_rlc_step(R, L, C, V0, t))))
    assert 3.0 < errs[0] / errs[1] < 5.0          # ≈ 4 for an O(h) scheme


def test_overdamped_closed_form_is_real_and_decays():
    t = np.linspace(0, 1e-2, 500)
    i = td.analytic_series_rlc_step(200.0, L, C, V0, t)
    assert np.all(np.isfinite(i)) and i[-1] < 1e-3 * i.max()


def test_talbot_matches_closed_forms():
    m = series_rlc("T")
    t = np.array([1e-4, 1e-3])
    v = td.step_voltage(m, [R, L, C], t, I0=1.0)      # L is dropped (acts at t = 0 only)
    assert np.allclose(v, R + t / C, rtol=1e-10)
    q = ImpedanceModel("Q0")
    a, Qv = 0.55, 0.02
    t = np.array([1e-2, 1.0, 10.0])
    v = td.step_voltage(q, [Qv, a], t)
    assert np.allclose(v, t ** a / (Qv * math.gamma(1 + a)), rtol=1e-10)


def test_talbot_vs_gl_for_parallel_rc():
    """R‖C under a current step: v = R I₀ (1 − e^{−t/RC})."""
    m = ImpedanceModel("p(R0,C0)")
    t = np.array([1e-4, 5e-4, 2e-3])
    v = td.step_voltage(m, [10.0, 1e-5], t)
    assert np.allclose(v, 10.0 * (1 - np.exp(-t / 1e-4)), rtol=1e-10)


def test_time_frequency_closure_for_fractional_rlc():
    """GL time simulation + lock-in recovers Z(iω) of the fractional model."""
    Rf, Lb, Qv, a, b = 5.0, 1e-3, 1e-6, 0.8, 0.95
    w0 = 2 * np.pi * 3000
    t = np.linspace(0, 0.01, 20001)
    g = td.gl_series_rlc(Rf, Lb, Qv, lambda x: np.sin(w0 * x), t, alpha=a, beta=b)
    Z_est = td.lockin_impedance(t, g["v"], g["i"], w0)
    Z_th = Rf + Lb * (1j * w0) ** b + 1 / (Qv * (1j * w0) ** a)
    assert abs(Z_est - Z_th) / abs(Z_th) < 1e-2
