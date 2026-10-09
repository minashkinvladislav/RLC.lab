"""Vector fields ``F: R³ → R³`` and their differential operators.

The driven series RLC is a non-autonomous planar system; adding the phase of
the source ``φ = ω t`` as a third coordinate makes it an autonomous system on
``R² × S¹`` – a vector field in three dimensions.  In the dimensionless
variables ``τ = ω₀ t``, ``x = q/q₀``, ``y = i/(ω₀ q₀)``::

    x' = y
    y' = −x − 2ζ y + u cos φ          ζ = (R/2)·sqrt(C/L),  ω₀ = 1/sqrt(LC)
    φ' = Ω                             Ω = ω/ω₀,  u = V₀ C / q₀

Then ``div F = −2ζ`` (uniform phase-volume contraction ⇒ Liouville:
``vol Φ_τ(A) = e^{−2ζτ} vol A``) and ``curl F = (u sin φ, 0, −2)``.

The :class:`VectorField3D` class keeps the field symbolically (SymPy) and
provides ``div``, ``curl``, the Jacobian, dot/cross products with other
fields, the Lie derivative of scalars, finite-difference counterparts for
verification and the flow-Jacobian determinant from variational equations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

__all__ = ["VectorField3D", "driven_rlc_field", "driven_rlc_field_dimensional", "gradient_field",
           "fd_divergence", "fd_curl"]


@dataclass
class VectorField3D:
    """Symbolic vector field ``F(x) = (F₁, F₂, F₃)`` on coordinates ``x``."""

    coords: Tuple[sp.Symbol, sp.Symbol, sp.Symbol]
    components: sp.Matrix
    params: Dict[sp.Symbol, float] | None = None
    name: str = "F"

    def __post_init__(self):
        self.components = sp.Matrix(self.components).reshape(3, 1)

    # ---- differential operators ----------------------------------------
    def divergence(self) -> sp.Expr:
        return sp.simplify(sum(sp.diff(self.components[i], self.coords[i]) for i in range(3)))

    def curl(self) -> sp.Matrix:
        F, (x, y, z) = self.components, self.coords
        return sp.simplify(sp.Matrix([
            sp.diff(F[2], y) - sp.diff(F[1], z),
            sp.diff(F[0], z) - sp.diff(F[2], x),
            sp.diff(F[1], x) - sp.diff(F[0], y),
        ]))

    def jacobian(self) -> sp.Matrix:
        return self.components.jacobian(sp.Matrix(self.coords))

    def dot(self, other: "VectorField3D | sp.Matrix") -> sp.Expr:
        G = other.components if isinstance(other, VectorField3D) else sp.Matrix(other)
        return sp.simplify((self.components.T * G)[0, 0])

    def cross(self, other: "VectorField3D | sp.Matrix") -> sp.Matrix:
        G = other.components if isinstance(other, VectorField3D) else sp.Matrix(other)
        return sp.simplify(self.components.cross(G))

    def lie_derivative(self, h: sp.Expr) -> sp.Expr:
        """``L_F h = ∇h · F`` – rate of change of ``h`` along trajectories."""
        grad = sp.Matrix([sp.diff(h, c) for c in self.coords])
        return sp.simplify((grad.T * self.components)[0, 0])

    # ---- numerics ------------------------------------------------------
    def numeric(self, params: Optional[Dict] = None) -> Callable:
        """``f(x, y, z) -> (F₁, F₂, F₃)`` vectorised over numpy arrays."""
        p = dict(self.params or {})
        p.update(params or {})
        comps = [sp.sympify(c).subs(p) for c in self.components]
        f = sp.lambdify(self.coords, comps, "numpy")

        def g(x, y, z):
            out = f(x, y, z)
            return [np.broadcast_to(np.asarray(o, dtype=float), np.shape(x)) for o in out]

        return g

    def numeric_expr(self, expr: sp.Expr, params: Optional[Dict] = None) -> Callable:
        p = dict(self.params or {})
        p.update(params or {})
        f = sp.lambdify(self.coords, sp.sympify(expr).subs(p), "numpy")
        return lambda x, y, z: np.broadcast_to(np.asarray(f(x, y, z), dtype=float), np.shape(x))

    def trajectory(self, x0: Sequence[float], t_span: Tuple[float, float], n: int = 2000,
                   params: Optional[Dict] = None, **kw):
        f = self.numeric(params)
        sol = solve_ivp(lambda t, s: [float(np.asarray(c)) for c in f(*s)], t_span, list(x0),
                        t_eval=np.linspace(*t_span, n), rtol=1e-9, atol=1e-12, **kw)
        return sol

    def flow_jacobian_det(self, x0: Sequence[float], T: float, params: Optional[Dict] = None) -> float:
        """``det DΦ_T(x0)`` from the variational equations ``Ṁ = DF(x(t)) M``.

        Liouville's formula gives ``det DΦ_T = exp(∫₀ᵀ div F(x(t)) dt)``.
        """
        p = dict(self.params or {})
        p.update(params or {})
        f = self.numeric(p)
        Jf = sp.lambdify(self.coords, self.jacobian().subs(p), "numpy")

        def rhs(t, s):
            x = s[:3]
            M = s[3:].reshape(3, 3)
            dx = [float(np.asarray(c)) for c in f(*x)]
            dM = np.asarray(Jf(*x), dtype=float) @ M
            return np.concatenate([dx, dM.ravel()])

        s0 = np.concatenate([np.asarray(x0, float), np.eye(3).ravel()])
        sol = solve_ivp(rhs, (0.0, T), s0, rtol=1e-10, atol=1e-12)
        return float(np.linalg.det(sol.y[3:, -1].reshape(3, 3)))


def driven_rlc_field(zeta: float | sp.Expr | None = None, u: float | sp.Expr | None = None,
                     Omega: float | sp.Expr | None = None) -> VectorField3D:
    """Dimensionless driven series RLC as an autonomous field on (x, y, φ)."""
    x, y, phi = sp.symbols("x y varphi", real=True)
    z_, u_, W_ = sp.symbols("zeta u Omega", positive=True)
    F = sp.Matrix([y, -x - 2 * z_ * y + u_ * sp.cos(phi), W_])
    params = {}
    for sym, val in ((z_, zeta), (u_, u), (W_, Omega)):
        if val is not None:
            params[sym] = val
    return VectorField3D((x, y, phi), F, params, name="driven RLC (dimensionless)")


def driven_rlc_field_dimensional() -> VectorField3D:
    """Same system in physical units on (q, i, φ): ``div F = −R/L``."""
    q, i, phi = sp.symbols("q i varphi", real=True)
    R, L, C, V0, w = sp.symbols("R L C V_0 omega", positive=True)
    F = sp.Matrix([i, (V0 * sp.cos(phi) - R * i - q / C) / L, w])
    return VectorField3D((q, i, phi), F, name="driven RLC")


def gradient_field(h: sp.Expr, coords: Tuple[sp.Symbol, sp.Symbol, sp.Symbol]) -> VectorField3D:
    return VectorField3D(coords, sp.Matrix([sp.diff(h, c) for c in coords]), name=f"grad {h}")


def fd_divergence(F: Callable, grid: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Second-order finite-difference divergence on a meshgrid (``indexing='ij'``)."""
    X, Y, Z = grid
    F1, F2, F3 = F(X, Y, Z)
    hx, hy, hz = X[1, 0, 0] - X[0, 0, 0], Y[0, 1, 0] - Y[0, 0, 0], Z[0, 0, 1] - Z[0, 0, 0]
    return (np.gradient(F1, hx, axis=0) + np.gradient(F2, hy, axis=1) + np.gradient(F3, hz, axis=2))


def fd_curl(F: Callable, grid: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, ...]:
    X, Y, Z = grid
    F1, F2, F3 = F(X, Y, Z)
    hx, hy, hz = X[1, 0, 0] - X[0, 0, 0], Y[0, 1, 0] - Y[0, 0, 0], Z[0, 0, 1] - Z[0, 0, 0]
    d = lambda A, h, ax: np.gradient(A, h, axis=ax)
    return (d(F3, hy, 1) - d(F2, hz, 2), d(F1, hz, 2) - d(F3, hx, 0), d(F2, hx, 0) - d(F1, hy, 1))
