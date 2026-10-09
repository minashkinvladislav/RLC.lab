"""Vector fields R^3 -> R^3: div, curl, identities, Liouville."""
import numpy as np
import sympy as sp

from rlcfreq import fields as F


def test_divergence_and_curl_of_driven_rlc():
    f = F.driven_rlc_field()
    x, y, phi = f.coords
    zeta, u, W = sp.symbols("zeta u Omega", positive=True)
    assert sp.simplify(f.divergence() + 2 * zeta) == 0
    assert sp.simplify(f.curl() - sp.Matrix([u * sp.sin(phi), 0, -2])) == sp.zeros(3, 1)


def test_dimensional_divergence_is_minus_r_over_l():
    f = F.driven_rlc_field_dimensional()
    R, L = sp.symbols("R L", positive=True)
    assert sp.simplify(f.divergence() + R / L) == 0


def test_vector_identities():
    f = F.driven_rlc_field()
    curl = F.VectorField3D(f.coords, f.curl())
    assert sp.simplify(curl.divergence()) == 0                      # div curl = 0
    h = f.coords[0] ** 2 * sp.sin(f.coords[2]) + f.coords[1] ** 3
    g = F.gradient_field(h, f.coords)
    assert sp.simplify(g.curl()) == sp.zeros(3, 1)                  # curl grad = 0


def test_power_balance_by_dot_product():
    """dH/dτ = ∇H·F = −2ζ y² + u y cos φ for H = (x² + y²)/2."""
    f = F.driven_rlc_field()
    x, y, phi = f.coords
    zeta, u = sp.symbols("zeta u", positive=True)
    H = (x ** 2 + y ** 2) / 2
    assert sp.simplify(f.lie_derivative(H) - (-2 * zeta * y ** 2 + u * y * sp.cos(phi))) == 0
    gH = F.gradient_field(H, f.coords)
    assert sp.simplify(f.dot(gH) - f.lie_derivative(H)) == 0
    cr = f.cross(gH)
    assert sp.simplify((cr.T * f.components)[0, 0]) == 0           # F×G ⟂ F


def test_finite_difference_operators_match_symbolic():
    zeta, u, W = 0.15, 0.8, 1.3
    f = F.driven_rlc_field(zeta, u, W)
    g = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21), np.linspace(0, 2 * np.pi, 41), indexing="ij")
    div = F.fd_divergence(f.numeric(), g)
    assert np.allclose(div, -2 * zeta, atol=1e-10)
    c1, c2, c3 = F.fd_curl(f.numeric(), g)
    inner = (slice(2, -2),) * 3
    assert np.allclose(c1[inner], u * np.sin(g[2][inner]), atol=5e-3)
    assert np.allclose(c3[inner], -2, atol=1e-10)


def test_liouville_volume_contraction():
    zeta = 0.2
    f = F.driven_rlc_field(zeta, 0.7, 0.9)
    T = 3.0
    det = f.flow_jacobian_det([0.3, -0.1, 0.0], T)
    assert np.isclose(det, np.exp(-2 * zeta * T), rtol=1e-7)
