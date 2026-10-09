"""Compiled models: values, Jacobians, transforms, reductions, canonical order."""
import numpy as np
import pytest

from rlcfreq import models


def _num_jac(model, theta, x, h=1e-7):
    cols = []
    for i in range(len(theta)):
        e = np.zeros_like(theta)
        e[i] = h * max(abs(theta[i]), 1e-12)
        cols.append((model.predict(theta + e, x) - model.predict(theta - e, x)) / (2 * e[i]))
    return np.stack(cols, axis=1)


@pytest.mark.parametrize("factory,kind", [(models.series_rlc, "T"), (models.series_rlc, "J"),
                                          (models.dummy_cell_model, "T"), (models.dummy_cell_model, "J+"),
                                          (models.battery_model, "T"), (models.battery_model, "J")])
def test_symbolic_jacobian_matches_finite_differences(factory, kind):
    m = factory(kind)
    th = m.expected()
    w = np.logspace(-1, 6, 25)
    Ja = m.jacobian(th, w)
    Jn = _num_jac(m, th, w)
    scale = np.abs(Jn).max(axis=0) + 1e-300
    assert np.max(np.abs(Ja - Jn) / scale) < 1e-5


def test_fractional_reduces_to_textbook():
    mt, mj = models.series_rlc("T"), models.series_rlc("J")
    w = np.logspace(1, 6, 50)
    R, L, C = 10.0, 1e-3, 1e-6
    zt = mt.predict([R, L, C], w)
    zj = mj.predict(mj.vector({"R0": R, "Lf0": L, "beta0": 1.0, "Q0": C, "alpha0": 1.0}), w)
    assert np.allclose(zt, zj, rtol=1e-12)


def test_series_rlc_resonance_and_minimum_modulus():
    m = models.series_rlc("T")
    R, L, C = 10.0, 1e-3, 1e-6
    w0 = 1 / np.sqrt(L * C)
    z = m.predict([R, L, C], np.array([w0]))[0]
    assert abs(z - R) < 1e-9


@pytest.mark.parametrize("transform,lo,hi,v", [("log", 1e-9, 1e3, 3.3e-4), ("logit", 0.3, 1.0, 0.77),
                                               ("lin", -1.0, 1.0, 0.25)])
def test_transforms_roundtrip_and_derivative(transform, lo, hi, v):
    s = models.ParameterSpec("x", v, lo, hi, transform)
    u = s.to_internal(v)
    assert np.isclose(s.from_internal(u), v)
    h = 1e-6
    num = (s.from_internal(u + h) - s.from_internal(u - h)) / (2 * h)
    assert np.isclose(num, s.derivative(v), rtol=1e-6)


def test_canonical_sorts_rc_blocks_by_time_constant():
    m = models.battery_model("T")
    th = m.vector(dict(L0=1e-7, R0=0.5, R1=0.3, C1=1.0, R2=0.1, C2=1e-3, C3=10.0))
    w = np.logspace(-2, 4, 40)
    th_c = m.canonical(th)
    assert np.allclose(m.predict(th, w), m.predict(th_c, w))       # invariance of Z
    taus = m.time_constants(th_c)
    assert taus == sorted(taus)


def test_transfer_function_only_for_rational_models():
    assert models.series_rlc("T").is_rational()
    with pytest.raises(ValueError):
        models.series_rlc("J").transfer_function()


def test_parameter_table_has_expected_values():
    tab = models.battery_model("J").table()
    assert {"expected", "lower", "upper", "unit"} <= set(tab.columns)
    assert (tab["lower"] <= tab["expected"]).all() and (tab["expected"] <= tab["upper"]).all()


def test_resonator_models_peak_at_resonance():
    for kind in ("T", "J"):
        m = models.resonator_model(kind, f_r=2e9)
        th = np.array([2e9, 80.0 if kind == "T" else 0.99, 0.01, 0.0, 0.0, 0.0])
        f = np.linspace(1.9e9, 2.1e9, 2001)
        s21 = m.predict(th, f)
        assert abs(f[np.argmax(np.abs(s21))] - 2e9) < 2e5
        assert np.isclose(np.abs(m.predict(th, np.array([2e9]))[0]), 0.01, rtol=1e-9)
