"""Symbolic derivations for the series RLC and its fractional variant.

* :func:`series_rlc_ode` – the textbook ODE ``L q̈ + R q̇ + q/C = v(t)``;
* :func:`impedance_from_ode` – ``Z(s) = V(s)/I(s)`` derived with SymPy's
  Laplace transform (zero initial conditions);
* :func:`fractional_series_rlc` – the fractional ODE
  ``L D^β i + R i + Q⁻¹ D^{−α} i = v`` and its impedance, using
  ``𝓛{D^γ f}(s) = s^γ F(s)`` (zero initial history);
* :func:`state_space` – first-order form used for the ODE graph;
* :func:`split_real_imag` – real and imaginary parts of ``Z(iω)`` for real
  ``ω > 0`` with the principal branch ``(iω)^γ = ω^γ e^{iγπ/2}``;
* :func:`resonance_frequency` – the unique zero of ``Im Z(iω)``;
* :func:`tf_report` – poles, natural frequency, damping and Q of a rational
  transfer function.
"""
from __future__ import annotations

from typing import Dict, Tuple

import sympy as sp

__all__ = ["series_rlc_ode", "impedance_from_ode", "fractional_series_rlc", "state_space",
           "split_real_imag", "resonance_frequency", "tf_report", "cpe_loss_tangent"]

t, s = sp.symbols("t s", positive=True)
omega = sp.Symbol("omega", positive=True)
R, L, C, Q = sp.symbols("R L C Q", positive=True)
alpha, beta = sp.symbols("alpha beta", positive=True)


def series_rlc_ode() -> Dict[str, sp.Basic]:
    q, v = sp.Function("q"), sp.Function("v")
    ode = sp.Eq(L * q(t).diff(t, 2) + R * q(t).diff(t) + q(t) / C, v(t))
    return dict(ode=ode, q=q, v=v, t=t, R=R, L=L, C=C)


def impedance_from_ode() -> sp.Expr:
    """Laplace-transform the ODE with ``q(0) = q̇(0) = 0``; return ``V/I``."""
    d = series_rlc_ode()
    q, v = d["q"], d["v"]
    lhs = sp.laplace_transform(d["ode"].lhs, t, s, noconds=True)
    Qs = sp.Symbol("Q_s")
    lhs = lhs.subs(sp.LaplaceTransform(q(t), t, s), Qs)
    lhs = lhs.subs({q(0): 0, sp.Subs(sp.Derivative(q(t), t), t, 0): 0})
    # V(s) = lhs(Q_s), I(s) = s Q_s  ->  Z = lhs / (s Q_s)
    return sp.simplify(sp.expand(lhs / (s * Qs)))


def fractional_series_rlc() -> Dict[str, sp.Basic]:
    """Fractional series RLC (Westerlund–Ekstam capacitor, Radwan–Salama inductor)."""
    Z = L * s ** beta + R + 1 / (Q * s ** alpha)
    text = r"L\,D^{\beta} i + R\,i + \frac{1}{Q} D^{-\alpha} i = v(t)"
    return dict(Z=Z, latex=text, reduces_to=sp.simplify(Z.subs({alpha: 1, beta: 1, Q: C})))


def state_space() -> Dict[str, sp.Expr]:
    """``q̇ = i``, ``i̇ = (v − R i − q/C)/L`` with symbols ``q, i, v``."""
    q, i, v = sp.symbols("q i v", real=True)
    return {"dq/dt": i, "di/dt": (v - R * i - q / C) / L, "symbols": (q, i, v)}


def split_real_imag(Zs: sp.Expr, s_sym: sp.Symbol = s, w: sp.Symbol = omega) -> Tuple[sp.Expr, sp.Expr]:
    """Real and imaginary part of ``Z(iω)`` for real positive ``ω``.

    Every power ``s^γ`` is replaced by ``ω^γ (cos(γπ/2) + i sin(γπ/2))``
    (principal branch); all parameters are assumed positive.
    """
    def repl(expr):
        if isinstance(expr, sp.Pow) and expr.base == s_sym:
            g = expr.exp
            return w ** g * (sp.cos(g * sp.pi / 2) + sp.I * sp.sin(g * sp.pi / 2))
        return None

    Zw = Zs.replace(lambda e: isinstance(e, sp.Pow) and e.base == s_sym, lambda e: repl(e))
    Zw = Zw.subs(s_sym, sp.I * w)
    re_, im_ = sp.expand_complex(Zw).as_real_imag()
    return sp.simplify(re_), sp.simplify(im_)


def resonance_frequency() -> Dict[str, sp.Expr]:
    """Zero of ``Im Z(iω)`` for the fractional series RLC.

    ``Im Z = L ω^β sin(βπ/2) − ω^{−α} sin(απ/2)/Q`` is strictly increasing for
    α, β ∈ (0, 1] and changes sign once, hence the unique root
    ``ω_r = [sin(απ/2) / (Q L sin(βπ/2))]^{1/(α+β)}``.
    """
    _, im_ = split_real_imag(fractional_series_rlc()["Z"])
    w_r = (sp.sin(alpha * sp.pi / 2) / (Q * L * sp.sin(beta * sp.pi / 2))) ** (1 / (alpha + beta))
    return dict(im=im_, omega_r=w_r, check=sp.simplify(sp.powdenest(im_.subs(omega, w_r), force=True)))


def cpe_loss_tangent() -> sp.Expr:
    """``tan δ = Re Z_CPE / (−Im Z_CPE) = cot(απ/2)`` – frequency independent."""
    re_, im_ = split_real_imag(1 / (Q * s ** alpha))
    return sp.simplify(re_ / (-im_))


def tf_report(tf) -> Dict[str, object]:
    """Poles/zeros of a SymPy ``TransferFunction``; ω₀, ζ, Q for 2nd-order factors."""
    poles = tf.poles()
    zeros = tf.zeros()
    den = sp.Poly(tf.den, tf.var)
    info: Dict[str, object] = dict(poles=poles, zeros=zeros, order=den.degree())
    if den.degree() == 2:
        a2, a1, a0 = den.all_coeffs()
        w0 = sp.sqrt(a0 / a2)
        zeta = sp.simplify(a1 / (2 * sp.sqrt(a0 * a2)))
        info.update(omega0=sp.simplify(w0), zeta=zeta, Q=sp.simplify(1 / (2 * zeta)))
    return info
