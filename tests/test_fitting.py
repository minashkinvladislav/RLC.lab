"""Parameter recovery on synthetic data with known truth."""
import numpy as np
import pytest

from rlcfreq import fitting, models


def _synthetic(model, theta, w, noise, seed=0):
    rng = np.random.default_rng(seed)
    z = model.predict(theta, w)
    return z * (1 + noise * (rng.standard_normal(w.size) + 1j * rng.standard_normal(w.size)) / np.sqrt(2))


@pytest.mark.parametrize("method", ["trf", "lm", "de", "multistart"])
def test_recover_series_rlc(method):
    m = models.series_rlc("T")
    truth = np.array([12.0, 2e-3, 0.5e-6])
    w = 2 * np.pi * np.logspace(2, 5, 60)
    y = _synthetic(m, truth, w, 1e-3)
    r = fitting.fit(m, w, y, method=method, seed=0)
    assert np.all(np.abs(r.theta - truth) / truth < 5 * r.rel_stderr + 1e-3)
    assert 0.5 < r.sigma2 / (1e-3 ** 2 / 2) < 2.0       # noise level recovered


def test_recover_fractional_dummy_model_and_coverage():
    m = models.dummy_cell_model("J")
    truth = m.vector(dict(R0=30.0, L0=3e-6, R1=47.0, Q1=1.05e-5, alpha1=0.995))
    w = 2 * np.pi * np.logspace(0, np.log10(5e4), 48)
    z = []
    for seed in range(20):
        y = _synthetic(m, truth, w, 5e-4, seed)
        start = truth * np.array([1.3, 0.7, 0.8, 1.5, 1.0]) - np.array([0, 0, 0, 0, 0.01])
        r = fitting.fit(m, w, y, theta0=start, method="trf")
        z.append((r.theta - truth) / r.stderr)
    z = np.array(z)
    # standardised errors ~ N(0, 1): Laplace approximation is calibrated
    assert np.all(np.abs(z.mean(axis=0)) < 0.8)
    assert np.all((0.5 < z.std(axis=0)) & (z.std(axis=0) < 1.6))


def test_bic_prefers_true_model():
    mt, mj = models.dummy_cell_model("T"), models.dummy_cell_model("J")
    truth = mj.vector(dict(R0=30.0, L0=3e-6, R1=47.0, Q1=1.05e-5, alpha1=0.98))
    w = 2 * np.pi * np.logspace(0, np.log10(5e4), 48)
    y = _synthetic(mj, truth, w, 5e-4)
    rt, rj = fitting.fit(mt, w, y, method="de"), fitting.fit(mj, w, y, method="de")
    assert rj.bic < rt.bic - 10


def test_tracking_follows_drifting_parameters():
    m = models.battery_model("J")
    base = m.expected()
    w = 2 * np.pi * np.logspace(-1.7, 4.3, 60)
    drift = np.linspace(1.0, 1.6, 12)
    names = m.names
    Y = []
    for d in drift:
        th = base.copy()
        th[names.index("R2")] *= d
        Y.append(_synthetic(m, th, w, 2e-4, seed=int(d * 100)))
    res = fitting.track(m, w, np.array(Y), base)
    r2 = np.array([r.params["R2"] for r in res])
    assert np.allclose(r2 / base[names.index("R2")], drift, rtol=0.02)


def test_summary_quantities_consistent():
    m = models.series_rlc("T")
    w = 2 * np.pi * np.logspace(2, 5, 40)
    y = _synthetic(m, np.array([12.0, 2e-3, 0.5e-6]), w, 1e-3)
    r = fitting.fit(m, w, y)
    assert r.n == 80 and r.k == 3
    assert np.isclose(r.rmse, np.sqrt(r.S / r.n))
    assert np.allclose(np.diag(r.corr), 1.0)
