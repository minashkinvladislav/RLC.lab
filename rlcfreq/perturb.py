"""Perturbations of data and of model structure (groundwork for labs 2–3).

Data perturbations (lab 2) – applied to complex spectra ``Z``:

* ``multiplicative`` – ``Z (1 + σ ε)``, ε circular standard normal (the noise
  model assumed by the modulus-weighted error function);
* ``additive``       – ``Z + σ·median|Z|·ε``;
* ``outliers``       – a fraction of points multiplied by ``1 + 10σ ε``;
* ``drift``          – smooth frequency-dependent bias ``Z (1 + σ log10(f/f_min))``,
  a crude model of instrument drift;
* ``frequency``      – jitter of the frequency axis (returned separately).

Model perturbations (lab 3) – :func:`structural_neighbours` enumerates
admissible circuit strings at distance one from a given one (replace
``C ↔ Q``, ``L ↔ Lf``, drop an element, add a series inductor).  Mixed models –
some elements ideal, some fractional – appear naturally in this neighbourhood;
ranking the neighbours by BIC is a one-step structure search over admissible
superpositions.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import circuit as cdsl

__all__ = ["perturb_data", "jitter_frequency", "structural_neighbours"]


def _circular(rng, n):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)


def perturb_data(Z: np.ndarray, kind: str = "multiplicative", level: float = 1e-3,
                 freq: Optional[np.ndarray] = None, fraction: float = 0.05, seed: int = 0) -> np.ndarray:
    """Return a perturbed copy of the complex spectrum ``Z``."""
    rng = np.random.default_rng(seed)
    Z = np.asarray(Z, dtype=complex)
    n = Z.size
    if kind == "multiplicative":
        return Z * (1 + level * _circular(rng, n))
    if kind == "additive":
        return Z + level * np.median(np.abs(Z)) * _circular(rng, n)
    if kind == "outliers":
        out = Z.copy()
        idx = rng.choice(n, size=max(1, int(round(fraction * n))), replace=False)
        out[idx] *= 1 + 10 * level * _circular(rng, idx.size)
        return out
    if kind == "drift":
        if freq is None:
            raise ValueError("drift needs the frequency axis")
        f = np.asarray(freq, dtype=float)
        return Z * (1 + level * np.log10(f / f.min()))
    raise ValueError(kind)


def jitter_frequency(freq: np.ndarray, level: float = 1e-3, seed: int = 0) -> np.ndarray:
    """Relative jitter of the frequency axis: ``f (1 + σ ε)``."""
    rng = np.random.default_rng(seed)
    return np.asarray(freq, dtype=float) * (1 + level * rng.standard_normal(len(freq)))


def _replace(node, target: cdsl.Element, new):
    if isinstance(node, cdsl.Element):
        return new if node == target else node
    kids = []
    for c in node.children:
        r = _replace(c, target, new)
        if r is not None:
            kids.append(r)
    if not kids:
        return None
    if len(kids) == 1:
        return kids[0]
    return type(node)(tuple(kids))


def structural_neighbours(text: str) -> List[Tuple[str, str]]:
    """Circuits at structural distance one: ``[(circuit, description), ...]``."""
    tree = cdsl.parse(text)
    elems = cdsl.elements(tree)
    used = {e.index for e in elems}
    out: List[Tuple[str, str]] = []
    swap = {"C": "Q", "Q": "C", "L": "Lf", "Lf": "L"}
    for e in elems:
        if e.kind in swap:
            new = cdsl.Element(swap[e.kind], e.index)
            if new.name not in {x.name for x in elems}:
                out.append((cdsl.to_string(_replace(tree, e, new)), f"{e.name} → {new.name}"))
        if len(elems) > 1:
            dropped = _replace(tree, e, None)
            if dropped is not None:
                s = cdsl.to_string(dropped)
                try:
                    cdsl.parse(s)
                    out.append((s, f"удалить {e.name}"))
                except ValueError:
                    pass
    if not any(e.kind in {"L", "Lf"} for e in elems):
        idx = next(str(i) for i in range(100) if str(i) not in used or f"L{i}" not in {x.name for x in elems})
        out.append((f"L{idx}-" + cdsl.to_string(tree), f"добавить L{idx}"))
    seen, uniq = set(), []
    for s, d in out:
        if s not in seen and s != cdsl.to_string(tree):
            seen.add(s)
            uniq.append((s, d))
    return uniq
