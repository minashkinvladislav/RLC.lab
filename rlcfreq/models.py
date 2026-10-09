"""Parametric models with an explicit parameter vector.

Every model exposes the same small interface used by :mod:`rlcfreq.fitting`:

* ``names``                      – free parameter names (the vector θ),
* ``specs[name]``                – :class:`ParameterSpec` with the expected value,
  admissible bounds, unit and the transform used by the optimisers,
* ``predict(theta, x)``          – complex model output,
* ``jacobian(theta, x)``         – ∂output/∂θ, complex, shape ``(len(x), k)``,
  obtained by symbolic differentiation (SymPy) and compiled with ``lambdify``.

Two families are provided:

:class:`ImpedanceModel`
    ``Z(iω; θ)`` of a circuit string (see :mod:`rlcfreq.circuit`); ``x = ω``.
:class:`ResonatorModel`
    Transmission ``S21(f; θ)`` of a weakly coupled resonator whose mode is a
    series RLC branch (textbook) or an L–CPE branch (fractional); ``x = f``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import sympy as sp

from . import circuit as cdsl

__all__ = [
    "ParameterSpec",
    "default_spec",
    "ImpedanceModel",
    "ResonatorModel",
    "series_rlc",
    "dummy_cell_model",
    "battery_model",
    "resonator_model",
]


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------
@dataclass
class ParameterSpec:
    """One coordinate of the parameter vector θ.

    ``transform`` defines the unconstrained internal coordinate ``u`` used by
    the optimisers: ``log`` (θ = eᵘ, θ > 0), ``logit`` (θ ∈ (lower, upper)) or
    ``lin`` (θ = u; bounds only bound the global search box).
    """

    name: str
    expected: float
    lower: float
    upper: float
    transform: str = "log"
    unit: str = ""
    description: str = ""

    def __post_init__(self):
        if self.transform not in {"log", "logit", "lin"}:
            raise ValueError(self.transform)
        if not (self.lower < self.upper):
            raise ValueError(f"{self.name}: lower >= upper")
        if self.transform == "log" and self.lower <= 0:
            raise ValueError(f"{self.name}: log transform needs lower > 0")

    # θ -> u
    def to_internal(self, v: float) -> float:
        if self.transform == "log":
            return math.log(v)
        if self.transform == "logit":
            a, b = self.lower, self.upper
            v = min(max(v, a + 1e-12 * (b - a)), b - 1e-12 * (b - a))
            return math.log((v - a) / (b - v))
        return float(v)

    # u -> θ  (overflow-safe: an optimiser may probe very large |u|)
    def from_internal(self, u: float) -> float:
        if self.transform == "log":
            return math.exp(min(max(u, -700.0), 700.0))
        if self.transform == "logit":
            a, b = self.lower, self.upper
            u = min(max(u, -700.0), 700.0)
            return a + (b - a) / (1.0 + math.exp(-u))
        return float(u)

    # dθ/du
    def derivative(self, v: float) -> float:
        if self.transform == "log":
            return v
        if self.transform == "logit":
            a, b = self.lower, self.upper
            return (v - a) * (b - v) / (b - a)
        return 1.0

    def search_box(self) -> tuple:
        """Box in internal coordinates for global optimisers."""
        if self.transform == "logit":
            return (-9.0, 9.0)          # θ within 1.2e-4 of the bounds
        return (self.to_internal(self.lower), self.to_internal(self.upper))


_DEFAULTS = {
    "R": dict(expected=1.0, lower=1e-6, upper=1e6, transform="log", unit="Ω", description="resistance"),
    "L": dict(expected=1e-6, lower=1e-12, upper=1e-1, transform="log", unit="H", description="inductance"),
    "C": dict(expected=1e-6, lower=1e-16, upper=1e3, transform="log", unit="F", description="capacitance"),
    "Q": dict(expected=1e-3, lower=1e-16, upper=1e4, transform="log", unit="F·s^(α−1)",
              description="CPE pseudo-capacitance"),
    "alpha": dict(expected=0.9, lower=0.3, upper=1.0, transform="logit", unit="—",
                  description="CPE exponent (α = 1: ideal capacitor)"),
    "Lf": dict(expected=1e-6, lower=1e-12, upper=1e-1, transform="log", unit="H·s^(β−1)",
               description="fractional inductance"),
    "beta": dict(expected=0.95, lower=0.5, upper=1.0, transform="logit", unit="—",
                 description="fractional-inductor exponent (β = 1: ideal inductor)"),
}

_NAME = re.compile(r"^(Lf|R|L|C|Q|alpha|beta)(\d+)$")


def default_spec(name: str, **overrides) -> ParameterSpec:
    """Default :class:`ParameterSpec` for a circuit parameter name."""
    m = _NAME.match(name)
    if m is None:
        raise ValueError(f"no default spec for {name!r}")
    kw = dict(_DEFAULTS[m.group(1)])
    kw.update(overrides)
    return ParameterSpec(name=name, **kw)


class _SymbolicModel:
    """Shared machinery: symbolic expression -> compiled value and Jacobian."""

    names: List[str]
    specs: Dict[str, ParameterSpec]
    fixed: Dict[str, float]
    _all_names: List[str]
    _symbols: Dict[str, sp.Symbol]
    _x: sp.Symbol
    expr: sp.Expr

    def _compile(self) -> None:
        args = [self._symbols[n] for n in self._all_names] + [self._x]
        self._f = sp.lambdify(args, self.expr, "numpy")
        self._df = None  # compiled lazily

    def _args(self, theta: Sequence[float]) -> List[float]:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (len(self.names),):
            raise ValueError(f"expected {len(self.names)} parameters, got shape {theta.shape}")
        values = dict(zip(self.names, theta))
        values.update(self.fixed)
        return [values[n] for n in self._all_names]

    # ---- parameter-vector helpers -------------------------------------
    @property
    def k(self) -> int:
        return len(self.names)

    def expected(self) -> np.ndarray:
        return np.array([self.specs[n].expected for n in self.names])

    def lower(self) -> np.ndarray:
        return np.array([self.specs[n].lower for n in self.names])

    def upper(self) -> np.ndarray:
        return np.array([self.specs[n].upper for n in self.names])

    def to_internal(self, theta: Sequence[float]) -> np.ndarray:
        return np.array([self.specs[n].to_internal(v) for n, v in zip(self.names, theta)])

    def from_internal(self, u: Sequence[float]) -> np.ndarray:
        return np.array([self.specs[n].from_internal(v) for n, v in zip(self.names, u)])

    def dtheta_du(self, theta: Sequence[float]) -> np.ndarray:
        return np.array([self.specs[n].derivative(v) for n, v in zip(self.names, theta)])

    def search_box(self) -> List[tuple]:
        return [self.specs[n].search_box() for n in self.names]

    def as_dict(self, theta: Sequence[float]) -> Dict[str, float]:
        return dict(zip(self.names, map(float, theta)))

    def vector(self, values: Mapping[str, float]) -> np.ndarray:
        """Parameter vector θ from a name -> value mapping."""
        return np.array([float(values[n]) for n in self.names])

    def table(self, theta: Optional[Sequence[float]] = None):
        """Parameter table (pandas) with expected values, bounds, units."""
        import pandas as pd

        rows = []
        for i, n in enumerate(self.names):
            sp_ = self.specs[n]
            row = dict(parameter=n, expected=sp_.expected, lower=sp_.lower, upper=sp_.upper,
                       transform=sp_.transform, unit=sp_.unit, description=sp_.description)
            if theta is not None:
                row["estimate"] = float(theta[i])
            rows.append(row)
        for n, v in self.fixed.items():
            rows.append(dict(parameter=n, expected=v, lower=v, upper=v, transform="fixed",
                             unit=self.specs[n].unit if n in self.specs else "", description="fixed",
                             **({"estimate": v} if theta is not None else {})))
        return pd.DataFrame(rows).set_index("parameter")

    # ---- evaluation ----------------------------------------------------
    def predict(self, theta: Sequence[float], x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = self._f(*self._args(theta), self._xvalue(x))
        return np.broadcast_to(np.asarray(out, dtype=complex), x.shape).copy()

    def jacobian(self, theta: Sequence[float], x: np.ndarray) -> np.ndarray:
        """Complex Jacobian ∂f/∂θ, shape ``(len(x), k)`` (symbolic derivatives)."""
        if self._df is None:
            args = [self._symbols[n] for n in self._all_names] + [self._x]
            derivs = [sp.diff(self.expr, self._symbols[n]) for n in self.names]
            self._df = sp.lambdify(args, derivs, "numpy")
        x = np.asarray(x, dtype=float)
        cols = self._df(*self._args(theta), self._xvalue(x))
        return np.stack([np.broadcast_to(np.asarray(c, dtype=complex), x.shape) for c in cols], axis=1)

    def _xvalue(self, x: np.ndarray):
        return x


# --------------------------------------------------------------------------
# impedance models
# --------------------------------------------------------------------------
class ImpedanceModel(_SymbolicModel):
    """Impedance ``Z(iω; θ)`` of a circuit string.

    Parameters
    ----------
    circuit:
        Circuit string, e.g. ``"R0-L0-p(R1,C1)"``.
    specs:
        Overrides of the default :class:`ParameterSpec` per parameter name
        (dicts of keyword arguments or ready ``ParameterSpec`` objects).
    fixed:
        Parameters held at fixed values (excluded from θ).
    name, description:
        Labels used in tables and plots.
    """

    def __init__(self, circuit: str, specs: Optional[Mapping[str, object]] = None,
                 fixed: Optional[Mapping[str, float]] = None, name: str = "",
                 description: str = ""):
        self.tree = cdsl.parse(circuit)
        self.circuit = cdsl.to_string(self.tree)
        self.name = name or self.circuit
        self.description = description
        self._all_names = cdsl.parameter_names(self.tree)
        self.fixed = {k: float(v) for k, v in (fixed or {}).items()}
        unknown = set(self.fixed) - set(self._all_names)
        if unknown:
            raise ValueError(f"fixed parameters not in circuit: {unknown}")
        self.names = [n for n in self._all_names if n not in self.fixed]
        self.specs: Dict[str, ParameterSpec] = {}
        for n in self._all_names:
            ov = (specs or {}).get(n, {})
            self.specs[n] = ov if isinstance(ov, ParameterSpec) else default_spec(n, **dict(ov))
        self.s = sp.Symbol("s")
        self._x = sp.Symbol("omega", positive=True)
        self._symbols = {n: cdsl.parameter_symbol(n) for n in self._all_names}
        self.expr_s = cdsl.impedance_expr(self.tree, self.s, self._symbols)
        self.expr = self.expr_s.subs(self.s, sp.I * self._x)
        self._compile()

    # Z(iω) is compiled with s = iω substituted; numpy uses the principal
    # branch (iω)^α = ω^α e^{iαπ/2}, which is the physical one for ω > 0.
    def impedance(self, theta: Sequence[float], omega: np.ndarray) -> np.ndarray:
        return self.predict(theta, omega)

    @property
    def symbols(self) -> Dict[str, sp.Symbol]:
        return dict(self._symbols)

    def is_rational(self) -> bool:
        """True if Z(s) is a rational function (integer-order ODE)."""
        return all(e.kind in {"R", "L", "C"} for e in cdsl.elements(self.tree))

    def with_fixed(self, **values) -> "ImpedanceModel":
        fx = dict(self.fixed)
        fx.update(values)
        return ImpedanceModel(self.circuit, specs=self.specs, fixed=fx, name=self.name,
                              description=self.description)

    def substituted(self, theta: Sequence[float], variable: str = "s") -> sp.Expr:
        """Expression with numeric parameter values (in ``s`` or ``omega``)."""
        values = dict(zip(self.names, map(float, theta)))
        values.update(self.fixed)
        expr = self.expr_s if variable == "s" else self.expr
        return expr.subs({self._symbols[n]: v for n, v in values.items()})

    def transfer_function(self, theta: Optional[Sequence[float]] = None, kind: str = "Z"):
        """SymPy ``TransferFunction`` of ``Z(s)`` or of the admittance ``Y = 1/Z``.

        Only for rational (integer-order) models.  For the series RLC the
        admittance ``I/V = C s / (L C s² + R C s + 1)`` is the proper
        second-order band-pass transfer function.
        """
        from sympy.physics.control.lti import TransferFunction

        if not self.is_rational():
            raise ValueError("Z(s) is not rational: the model contains fractional elements")
        expr = self.expr_s if theta is None else self.substituted(theta, "s")
        if kind == "Y":
            expr = 1 / expr
        num, den = sp.fraction(sp.together(expr))
        return TransferFunction(sp.expand(num), sp.expand(den), self.s)

    def mpmath_function(self, theta: Sequence[float], drop_inductive: bool = True) -> Callable:
        """``Z(s)`` as an mpmath-callable (for numerical inverse Laplace).

        With ``drop_inductive`` the ``L`` / ``Lf`` elements are set to zero:
        their contribution to the response to a current *step* is concentrated
        at ``t = 0`` (``L di/dt = L I₀ δ(t)``) and they are irrelevant for
        ``t > 0``.  Keeping them would make ``Z(s)/s`` non-decaying, which the
        Talbot contour does not tolerate.
        """
        values = dict(zip(self.names, map(float, theta)))
        values.update(self.fixed)
        if drop_inductive:
            for e in cdsl.elements(self.tree):
                if e.kind in {"L", "Lf"}:
                    values[e.params[0]] = 0.0
        expr = self.expr_s.subs({self._symbols[n]: v for n, v in values.items()})
        return sp.lambdify(self.s, expr, "mpmath")

    def rc_blocks(self) -> List[tuple]:
        """Parallel blocks of the form ``p(Rk, Ck)`` / ``p(Rk, Qk)``.

        Returns ``[(R_name, X_name, alpha_name_or_None), ...]``.  Such blocks
        in series are interchangeable (Z is invariant under their
        permutation), which makes the parameter vector identifiable only up to
        a permutation – see :meth:`canonical`.
        """
        blocks = []

        def walk(n):
            if isinstance(n, cdsl.Parallel):
                kids = n.children
                if len(kids) == 2 and all(isinstance(c, cdsl.Element) for c in kids):
                    kinds = {c.kind: c for c in kids}
                    if "R" in kinds and ("C" in kinds or "Q" in kinds):
                        x = kinds.get("C") or kinds.get("Q")
                        blocks.append((kinds["R"].params[0], x.params[0],
                                       x.params[1] if x.kind == "Q" else None))
                        return
            if not isinstance(n, cdsl.Element):
                for c in n.children:
                    walk(c)

        walk(self.tree)
        return blocks

    def time_constants(self, theta: Sequence[float]) -> List[float]:
        """Characteristic times τ = (R·Q)^{1/α} (τ = R·C for α = 1) of RC blocks."""
        v = dict(zip(self.names, map(float, theta)))
        v.update(self.fixed)
        out = []
        for r, x, a in self.rc_blocks():
            alpha = v[a] if a else 1.0
            out.append((v[r] * v[x]) ** (1.0 / alpha))
        return out

    def canonical(self, theta: Sequence[float]) -> np.ndarray:
        """Sort interchangeable RC blocks by increasing time constant.

        Only blocks with the same element kinds are permuted, and only if all
        their parameters are free.  Removes the label-switching symmetry.
        """
        theta = np.asarray(theta, dtype=float).copy()
        blocks = [b for b in self.rc_blocks() if all(p is None or p in self.names for p in b)]
        if len(blocks) < 2:
            return theta
        groups: Dict[bool, List[tuple]] = {}
        for b in blocks:
            groups.setdefault(b[2] is None, []).append(b)
        idx = {n: i for i, n in enumerate(self.names)}
        for grp in groups.values():
            if len(grp) < 2:
                continue
            taus = []
            for r, x, a in grp:
                alpha = theta[idx[a]] if a else 1.0
                taus.append((theta[idx[r]] * theta[idx[x]]) ** (1.0 / alpha))
            order = np.argsort(taus)
            values = [[theta[idx[p]] for p in b if p is not None] for b in grp]
            for target, src in zip(grp, order):
                for p, v in zip([p for p in target if p is not None], values[src]):
                    theta[idx[p]] = v
        return theta

    def __repr__(self) -> str:
        return f"ImpedanceModel({self.circuit!r}, k={self.k}, name={self.name!r})"


# --------------------------------------------------------------------------
# transmission resonator
# --------------------------------------------------------------------------
class ResonatorModel(_SymbolicModel):
    """S21 of a weakly coupled resonator mode around its resonance.

    The mode is a two-terminal branch with normalised impedance
    ``z(x) = Z / (ω_r L)``, ``x = f / f_r``:

    * ``kind='T'`` (textbook series RLC):  ``z = 1/Q + i (x − 1/x)``;
    * ``kind='J'`` (inductor + CPE, no separate resistor):
      ``z = i x + (i x)^(−α) / sin(απ/2)``, whose loss is entirely due to the
      constant-phase element; at resonance ``z = cot(απ/2) = tan δ``.

    Weak coupling makes ``S21`` proportional to the branch admittance, plus a
    frequency-independent leakage ``d``:  ``S21 = g · z(1) / z(x) + d`` with
    complex ``g`` (peak transmission) and ``d``.  θ = (f_r, Q or α, Re g, Im g,
    Re d, Im d).
    """

    def __init__(self, kind: str = "T", f_r: float = 2e9, f_span: float = 1e8, name: str = ""):
        if kind not in {"T", "J"}:
            raise ValueError(kind)
        self.kind = kind
        self.name = name or f"resonator-{kind}"
        x, fr, Q, a = sp.symbols("x f_r Q alpha", positive=True)
        gr, gi, dr, di = sp.symbols("g_re g_im d_re d_im", real=True)
        f = sp.Symbol("f", positive=True)
        xx = f / fr
        if kind == "T":
            z = 1 / Q + sp.I * (xx - 1 / xx)
            z1 = 1 / Q
            shape = ("Q", Q, dict(expected=50.0, lower=1.0, upper=1e5, transform="log", unit="—",
                                   description="quality factor (loaded)"))
        else:
            z = sp.I * xx + (sp.I * xx) ** (-a) / sp.sin(a * sp.pi / 2)
            z1 = sp.cot(a * sp.pi / 2)
            shape = ("alpha", a, dict(expected=0.98, lower=0.8, upper=1.0, transform="logit", unit="—",
                                       description="CPE exponent of the mode capacitance"))
        self.z_expr = z
        self.expr = (gr + sp.I * gi) * z1 / z + (dr + sp.I * di)
        self._x = f
        sname, ssym, sspec = shape
        self._symbols = {"f_r": fr, sname: ssym, "g_re": gr, "g_im": gi, "d_re": dr, "d_im": di}
        self._all_names = ["f_r", sname, "g_re", "g_im", "d_re", "d_im"]
        self.names = list(self._all_names)
        self.fixed = {}
        self.specs = {
            "f_r": ParameterSpec("f_r", f_r, f_r - f_span, f_r + f_span, "lin", "Hz", "resonance frequency"),
            sname: ParameterSpec(sname, **sspec),
            "g_re": ParameterSpec("g_re", 0.0, -1, 1, "lin", "—", "peak transmission (real part)"),
            "g_im": ParameterSpec("g_im", 0.0, -1, 1, "lin", "—", "peak transmission (imag. part)"),
            "d_re": ParameterSpec("d_re", 0.0, -1, 1, "lin", "—", "leakage (real part)"),
            "d_im": ParameterSpec("d_im", 0.0, -1, 1, "lin", "—", "leakage (imag. part)"),
        }
        self._compile()

    def q_factor(self, theta: Sequence[float]) -> float:
        """Loaded Q: the parameter itself (T) or tan(απ/2) (J)."""
        return float(theta[1]) if self.kind == "T" else float(np.tan(theta[1] * np.pi / 2))

    def __repr__(self) -> str:
        return f"ResonatorModel(kind={self.kind!r})"


# --------------------------------------------------------------------------
# model catalogue used in the lab
# --------------------------------------------------------------------------
def series_rlc(kind: str = "T") -> ImpedanceModel:
    """Series RLC: textbook ``R0-L0-C0`` or fractional ``R0-Lf0-Q0``.

    Expected values are those of a generic laboratory RLC (R = 10 Ω,
    L = 1 mH, C = 1 µF; ω₀ = 31.6 krad/s).
    """
    if kind == "T":
        return ImpedanceModel("R0-L0-C0", name="series RLC (textbook)",
                              specs={"R0": dict(expected=10.0), "L0": dict(expected=1e-3),
                                     "C0": dict(expected=1e-6)})
    return ImpedanceModel("R0-Lf0-Q0", name="series RLC (fractional)",
                          specs={"R0": dict(expected=10.0), "Lf0": dict(expected=1e-3),
                                 "Q0": dict(expected=1e-6), "alpha0": dict(expected=0.95),
                                 "beta0": dict(expected=0.95)})


def dummy_cell_model(kind: str = "T") -> ImpedanceModel:
    """Discrete-component test circuit ``R0 - L0 - (R1 ‖ C1)``.

    ``T``: ideal elements; ``J``: the capacitor replaced by a CPE (Westerlund
    capacitor); ``J+``: additionally a fractional inductor (Radwan–Salama).
    The lead inductance ``L0`` is the parasitic of the test fixture.
    """
    base = {"R0": dict(expected=100.0, lower=1e-2, upper=1e5),
            "R1": dict(expected=500.0, lower=1e-2, upper=1e6),
            "L0": dict(expected=1e-6, lower=1e-10, upper=1e-2, description="lead inductance"),
            "Lf0": dict(expected=1e-6, lower=1e-10, upper=1e-2, description="lead inductance (fractional)"),
            "C1": dict(expected=1e-7, lower=1e-13, upper=1e-2),
            "Q1": dict(expected=1e-7, lower=1e-13, upper=1e-2),
            "alpha1": dict(expected=0.99, lower=0.8, upper=1.0),
            "beta0": dict(expected=0.99, lower=0.8, upper=1.0)}
    circ = {"T": "R0-L0-p(R1,C1)", "J": "R0-L0-p(R1,Q1)", "J+": "R0-Lf0-p(R1,Q1)"}[kind]
    names = {"T": "test circuit (textbook)", "J": "test circuit (CPE)", "J+": "test circuit (CPE + frac. L)"}
    m = ImpedanceModel(circ, name=names[kind])
    specs = {n: base[n] for n in m.names}
    return ImpedanceModel(circ, specs=specs, name=names[kind])


def battery_model(kind: str = "T") -> ImpedanceModel:
    """Equivalent circuit of a Li-ion cell, ``L0-R0-(R1‖·)-(R2‖·)-·``.

    ``T`` (textbook, integer order): ``L0-R0-p(R1,C1)-p(R2,C2)-C3``;
    ``J`` (CPE, journal): ``L0-R0-p(R1,Q1)-p(R2,Q2)-Q3``.
    The last element describes the low-frequency (solid-state diffusion /
    intercalation) branch; for α₃ = 1/2 the CPE is a semi-infinite Warburg
    element.  Expected values: a 45 mAh coin cell, R₀ ≈ 0.5 Ω, arcs of a few
    tenths of an ohm.
    """
    common = {"L0": dict(expected=3e-7, lower=1e-10, upper=1e-4),
              "R0": dict(expected=0.5, lower=1e-3, upper=5.0),
              "R1": dict(expected=0.2, lower=1e-4, upper=5.0),
              "R2": dict(expected=0.2, lower=1e-4, upper=5.0)}
    if kind == "T":
        specs = dict(common, C1=dict(expected=1e-3, lower=1e-7, upper=1.0),
                     C2=dict(expected=1e-1, lower=1e-6, upper=1e2),
                     C3=dict(expected=10.0, lower=1e-3, upper=1e5))
        return ImpedanceModel("L0-R0-p(R1,C1)-p(R2,C2)-C3", specs=specs, name="Li-ion cell (textbook RC)")
    specs = dict(common, Q1=dict(expected=3e-2, lower=1e-7, upper=1e2),
                 alpha1=dict(expected=0.6, lower=0.3, upper=1.0),
                 Q2=dict(expected=3e-2, lower=1e-6, upper=1e2),
                 alpha2=dict(expected=0.9, lower=0.3, upper=1.0),
                 Q3=dict(expected=10.0, lower=1e-3, upper=1e5),
                 alpha3=dict(expected=0.6, lower=0.3, upper=1.0))
    return ImpedanceModel("L0-R0-p(R1,Q1)-p(R2,Q2)-Q3", specs=specs, name="Li-ion cell (CPE)")


def resonator_model(kind: str = "T", f_r: float = 2e9, f_span: float = 1e8) -> ResonatorModel:
    return ResonatorModel(kind=kind, f_r=f_r, f_span=f_span,
                          name="stripline mode (series RLC)" if kind == "T" else "stripline mode (L + CPE)")
