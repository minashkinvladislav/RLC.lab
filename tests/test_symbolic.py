"""Symbolic derivations: ODE -> Laplace -> Z, fractional variant, resonance."""
import numpy as np
import sympy as sp

from rlcfreq import symbolic as S
from rlcfreq.models import series_rlc


def test_impedance_from_ode_equals_textbook_formula():
    Z = S.impedance_from_ode()
    assert sp.simplify(Z - (S.R + S.L * S.s + 1 / (S.C * S.s))) == 0


def test_fractional_model_reduces_for_unit_orders():
    d = S.fractional_series_rlc()
    assert sp.simplify(d["reduces_to"] - S.impedance_from_ode()) == 0


def test_real_imag_split_numerically():
    Z = S.fractional_series_rlc()["Z"]
    re_, im_ = S.split_real_imag(Z)
    vals = {S.R: 3.0, S.L: 2e-3, S.Q: 4e-6, S.alpha: 0.7, S.beta: 0.9}
    for w in (10.0, 1e3, 1e5):
        z = complex(Z.subs(vals).subs(S.s, 1j * w))
        assert np.isclose(float(re_.subs(vals).subs(S.omega, w)), z.real)
        assert np.isclose(float(im_.subs(vals).subs(S.omega, w)), z.imag)


def test_resonance_frequency_formula():
    r = S.resonance_frequency()
    assert r["check"] == 0
    # α = β = 1 gives 1/sqrt(LC)
    w = r["omega_r"].subs({S.alpha: 1, S.beta: 1, S.Q: S.C})
    assert sp.simplify(w - 1 / sp.sqrt(S.L * S.C)) == 0


def test_cpe_loss_tangent_is_cot():
    assert sp.simplify(S.cpe_loss_tangent() - sp.cot(S.alpha * sp.pi / 2)) == 0


def test_tf_report_series_rlc():
    rep = S.tf_report(series_rlc("T").transfer_function(kind="Y"))
    R, L, C = (series_rlc("T").symbols[n] for n in ("R0", "L0", "C0"))
    assert sp.simplify(rep["omega0"] - 1 / sp.sqrt(L * C)) == 0
    assert sp.simplify(rep["zeta"] - R / 2 * sp.sqrt(C / L)) == 0
    assert rep["order"] == 2
