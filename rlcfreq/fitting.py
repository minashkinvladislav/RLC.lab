"""Parameter estimation for complex-valued frequency-domain models.

Error function
--------------
For complex observations ``y_k`` and model values ``ŷ_k(θ)`` the residual is

    r_k(θ) = w_k (ŷ_k(θ) − y_k),        S(θ) = Σ_k |r_k(θ)|²,

with ``w_k = 1/|y_k|`` ("modulus weighting", the standard choice in impedance
spectroscopy, Orazem & Tribollet 2008) or a constant weight.  Real and
imaginary parts are stacked, so there are ``n = 2N`` real residuals.  Under the
assumption ``y_k = ŷ_k(θ*) + ε_k`` with independent circular Gaussian errors
of standard deviation ``σ/w_k`` minimising ``S`` is maximum likelihood.

Optimisers
----------
``trf``        trust-region reflective Levenberg–Marquardt (SciPy) with the
               analytic (symbolic) Jacobian,
``lm``         MINPACK Levenberg–Marquardt (unconstrained internal coordinates),
``de``         differential evolution (global) followed by ``trf`` polishing,
``cmaes``      CMA-ES (``cma`` package, optional) followed by ``trf``,
``multistart`` several ``trf`` runs from Latin-hypercube starting points.

All optimisers work in the unconstrained internal coordinates ``u`` defined
by each :class:`~rlcfreq.models.ParameterSpec` (log / logit / linear).

Uncertainty
-----------
Laplace (Gauss–Newton) approximation at the optimum:
``Cov(θ̂) ≈ σ̂² (JᵀJ)⁺``, ``σ̂² = S(θ̂)/(n − k)``, with ``J = ∂r/∂θ``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.stats import qmc

__all__ = [
    "FitResult",
    "make_weights",
    "residuals",
    "summarize",
    "fit",
    "track",
    "compare_optimizers",
]

Weights = Union[str, np.ndarray]


def make_weights(y: np.ndarray, weights: Weights = "modulus") -> np.ndarray:
    y = np.asarray(y)
    if isinstance(weights, str):
        if weights == "modulus":
            return 1.0 / np.abs(y)
        if weights == "unit":
            return np.full(y.shape, 1.0 / np.max(np.abs(y)))
        if weights == "none":
            return np.ones(y.shape)
        raise ValueError(weights)
    w = np.asarray(weights, dtype=float)
    if w.shape != y.shape:
        raise ValueError("weights must have the shape of y")
    return w


def residuals(model, theta, x, y, w) -> np.ndarray:
    """Stacked real residual vector ``[Re r, Im r]`` of length 2N."""
    with np.errstate(all="ignore"):
        r = (model.predict(theta, x) - y) * w
    out = np.concatenate([r.real, r.imag])
    return np.nan_to_num(out, nan=1e10, posinf=1e10, neginf=-1e10)


def _jac_theta(model, theta, x, w) -> np.ndarray:
    with np.errstate(all="ignore"):
        Jc = model.jacobian(theta, x) * w[:, None]
    return np.nan_to_num(np.vstack([Jc.real, Jc.imag]), nan=0.0, posinf=1e10, neginf=-1e10)


@dataclass
class FitResult:
    """Outcome of one fit, with error measures and a covariance estimate."""

    model_name: str
    names: List[str]
    theta: np.ndarray
    S: float                    # weighted residual sum of squares
    n: int                      # number of real residuals (2N)
    k: int                      # number of free parameters
    stderr: np.ndarray
    cov: np.ndarray
    corr: np.ndarray
    cond: float                 # condition number of the column-normalised Jacobian
    method: str = ""
    nfev: int = 0
    seconds: float = 0.0
    success: bool = True
    message: str = ""
    residual: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)

    @property
    def rmse(self) -> float:
        """sqrt(S/n); with modulus weighting: RMS relative error."""
        return float(np.sqrt(self.S / self.n))

    @property
    def sigma2(self) -> float:
        return float(self.S / max(self.n - self.k, 1))

    @property
    def aic(self) -> float:
        return float(self.n * np.log(self.S / self.n) + 2 * self.k)

    @property
    def bic(self) -> float:
        return float(self.n * np.log(self.S / self.n) + self.k * np.log(self.n))

    @property
    def params(self) -> Dict[str, float]:
        return dict(zip(self.names, map(float, self.theta)))

    @property
    def rel_stderr(self) -> np.ndarray:
        return self.stderr / np.abs(self.theta)

    def summary(self) -> Dict[str, float]:
        return dict(model=self.model_name, method=self.method, k=self.k, n=self.n, S=self.S,
                    rmse=self.rmse, aic=self.aic, bic=self.bic, cond=self.cond, nfev=self.nfev,
                    seconds=self.seconds)

    def table(self):
        import pandas as pd

        return pd.DataFrame({"estimate": self.theta, "stderr": self.stderr,
                             "rel_stderr_%": 100 * self.rel_stderr}, index=self.names)


def summarize(model, theta, x, y, w, method="", nfev=0, seconds=0.0, success=True,
              message="") -> FitResult:
    """Error measures and the Laplace-approximation covariance at θ."""
    theta = np.asarray(theta, dtype=float)
    r = residuals(model, theta, x, y, w)
    S = float(r @ r)
    n, k = r.size, len(theta)
    J = _jac_theta(model, theta, x, w)
    # collinearity diagnostic (Belsley): condition number of the Jacobian with
    # columns scaled to unit norm – invariant to units and parameter scaling
    norms = np.linalg.norm(J, axis=0)
    Js = J / np.where(norms > 0, norms, 1.0)[None, :]
    sv = np.linalg.svd(Js, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    sigma2 = S / max(n - k, 1)
    # Cov = σ² (JᵀJ)⁺ computed stably via the SVD of the column-scaled J:
    # with J = Js·D, (JᵀJ)⁺ = D⁻¹ V Σ⁻² Vᵀ D⁻¹  (never forms the ill-scaled JᵀJ)
    _, svals, Vt = np.linalg.svd(Js, full_matrices=False)
    keep = svals > svals[0] * 1e-12
    inv = (Vt[keep].T / svals[keep] ** 2) @ Vt[keep]
    dinv = 1.0 / np.where(norms > 0, norms, 1.0)
    cov = sigma2 * inv * np.outer(dinv, dinv)
    stderr = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(stderr, stderr)
    return FitResult(getattr(model, "name", type(model).__name__), list(model.names), theta, S, n, k,
                     stderr, cov, corr, cond, method, nfev, seconds, success, message, r)


def _local(model, x, y, w, u0, method="trf", jac="analytic", max_nfev=2000):
    def fun(u):
        return residuals(model, model.from_internal(u), x, y, w)

    def jfun(u):
        th = model.from_internal(u)
        return _jac_theta(model, th, x, w) * model.dtheta_du(th)[None, :]

    kw = dict(method=method, x_scale="jac", max_nfev=max_nfev, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    if jac == "analytic":
        kw["jac"] = jfun
    with np.errstate(all="ignore"):
        return least_squares(fun, np.asarray(u0, dtype=float), **kw)


def fit(model, x, y, theta0: Optional[Sequence[float]] = None, weights: Weights = "modulus",
        method: str = "trf", jac: str = "analytic", seed: int = 0, de_maxiter: int = 400,
        n_starts: int = 16, popsize: int = 20, polish: bool = True) -> FitResult:
    """Fit ``model`` to complex data ``y`` observed at ``x``.

    ``method`` is one of ``trf``, ``lm``, ``de``, ``cmaes``, ``multistart``
    (see module docstring).  Returns a :class:`FitResult`.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=complex)
    w = make_weights(y, weights)
    theta0 = model.expected() if theta0 is None else np.asarray(theta0, dtype=float)
    u0 = model.to_internal(theta0)
    t0 = time.perf_counter()
    nfev = 0

    def S_of_u(u):
        r = residuals(model, model.from_internal(u), x, y, w)
        v = float(r @ r)
        return v if np.isfinite(v) else 1e300

    if method in {"trf", "lm"}:
        res = _local(model, x, y, w, u0, method=method, jac=jac)
        u, nfev, ok, msg = res.x, res.nfev, res.success, res.message
    elif method == "de":
        with np.errstate(all="ignore"):
            de = differential_evolution(S_of_u, model.search_box(), seed=seed, maxiter=de_maxiter,
                                        popsize=popsize, tol=1e-12, polish=False, init="sobol",
                                        x0=np.clip(u0, *np.array(model.search_box()).T))
        u, nfev, ok, msg = de.x, de.nfev, True, "differential evolution"
        if polish:
            res = _local(model, x, y, w, u, jac=jac)
            if res.cost * 2 <= de.fun:
                u = res.x
            nfev += res.nfev
    elif method == "cmaes":
        import cma  # optional dependency

        box = np.array(model.search_box())
        sigma0 = 0.25 * float(np.median(box[:, 1] - box[:, 0]))
        es = cma.CMAEvolutionStrategy(np.clip(u0, box[:, 0], box[:, 1]), sigma0,
                                      {"bounds": [list(box[:, 0]), list(box[:, 1])], "seed": seed + 1,
                                       "verbose": -9, "maxfevals": 40000, "tolfun": 1e-14,
                                       "CMA_stds": list((box[:, 1] - box[:, 0]) / np.median(box[:, 1] - box[:, 0]))})
        es.optimize(S_of_u)
        u, nfev, ok, msg = es.result.xbest, es.result.evaluations, True, "CMA-ES"
        if polish:
            res = _local(model, x, y, w, u, jac=jac)
            if res.cost * 2 <= es.result.fbest:
                u = res.x
            nfev += res.nfev
    elif method == "multistart":
        box = np.array(model.search_box())
        starts = qmc.scale(qmc.LatinHypercube(d=len(u0), seed=seed).random(n_starts), box[:, 0], box[:, 1])
        best = _local(model, x, y, w, u0, jac=jac)
        nfev = best.nfev
        for s0 in starts:
            res = _local(model, x, y, w, s0, jac=jac)
            nfev += res.nfev
            if res.cost < best.cost:
                best = res
        u, ok, msg = best.x, best.success, f"best of {n_starts + 1} starts"
    else:
        raise ValueError(f"unknown method {method!r}")
    theta = model.from_internal(u)
    if hasattr(model, "canonical"):
        theta = model.canonical(theta)
    return summarize(model, theta, x, y, w, method=method, nfev=int(nfev),
                     seconds=time.perf_counter() - t0, success=bool(ok), message=str(msg))


def track(model, x, Y: np.ndarray, theta0: Sequence[float], weights: Weights = "modulus",
          method: str = "trf", refit_every: int = 0, refit_method: str = "de",
          callback: Optional[Callable[[int, FitResult], None]] = None,
          checks: Optional[list] = None) -> List[FitResult]:
    """Fit a sequence of spectra ``Y[j]`` with warm starts (continuation).

    The solution for spectrum ``j`` initialises spectrum ``j+1`` – the
    parameters of an ageing cell drift slowly, so the previous optimum lies in
    the basin of attraction of the next one.  With ``refit_every > 0`` a
    global optimisation is run every ``refit_every`` spectra and the better of
    the two solutions is kept (guards against tracking a local minimum); if
    a list is passed as ``checks``, each comparison is appended to it as
    ``dict(index, S_warm, S_global)``.
    """
    out: List[FitResult] = []
    theta = np.asarray(theta0, dtype=float)
    for j, y in enumerate(np.asarray(Y)):
        r = fit(model, x, y, theta0=theta, weights=weights, method=method)
        if refit_every and j % refit_every == 0:
            g = fit(model, x, y, theta0=theta, weights=weights, method=refit_method)
            if checks is not None:
                checks.append(dict(index=j, S_warm=r.S, S_global=g.S))
            if g.S < r.S:
                r = g
        out.append(r)
        theta = r.theta
        if callback is not None:
            callback(j, r)
    return out


def compare_optimizers(model, x, y, theta0=None, weights: Weights = "modulus",
                       methods: Sequence[str] = ("trf", "lm", "multistart", "de", "cmaes"),
                       seed: int = 0):
    """Run several optimisers on the same problem; returns a pandas table."""
    import pandas as pd

    rows, results = [], {}
    for m in methods:
        try:
            r = fit(model, x, y, theta0=theta0, weights=weights, method=m, seed=seed)
        except ImportError:
            continue
        results[m] = r
        rows.append(dict(method=m, S=r.S, rmse=r.rmse, nfev=r.nfev, seconds=r.seconds,
                         **{n: v for n, v in r.params.items()}))
    return pd.DataFrame(rows).set_index("method"), results
