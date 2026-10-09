"""Perturbations of data and structure."""
import numpy as np
import pytest

from rlcfreq import circuit, perturb


def test_multiplicative_noise_level():
    Z = np.full(20000, 3.0 + 4.0j)
    Zp = perturb.perturb_data(Z, "multiplicative", 1e-2, seed=1)
    rel = (Zp - Z) / Z
    assert np.isclose(np.sqrt(np.mean(np.abs(rel) ** 2)), 1e-2, rtol=0.03)


def test_outliers_fraction():
    Z = np.ones(200, complex)
    Zp = perturb.perturb_data(Z, "outliers", 1e-2, fraction=0.05, seed=0)
    assert np.sum(Zp != Z) == 10


def test_drift_is_deterministic_and_zero_at_fmin():
    f = np.logspace(0, 4, 9)
    Zp = perturb.perturb_data(np.ones(9, complex), "drift", 0.01, freq=f)
    assert Zp[0] == 1 and np.isclose(Zp[-1], 1.04)


def test_neighbours_are_valid_and_include_mixed_models():
    nb = perturb.structural_neighbours("L0-R0-p(R1,C1)-p(R2,C2)-C3")
    names = [s for s, _ in nb]
    for s in names:
        circuit.parse(s)                       # every neighbour is admissible
    assert "L0-R0-p(R1,Q1)-p(R2,C2)-C3" in names    # mixed ideal/fractional model
    assert "Lf0-R0-p(R1,C1)-p(R2,C2)-C3" in names
    assert "L0-R0-p(R1,C1)-p(R2,C2)" in names        # element removed
    assert len(set(names)) == len(names)


def test_neighbours_add_inductor_when_missing():
    names = [s for s, _ in perturb.structural_neighbours("R0-p(R1,C1)")]
    assert any(s.startswith("L") for s in names)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        perturb.perturb_data(np.ones(3), "nope")
