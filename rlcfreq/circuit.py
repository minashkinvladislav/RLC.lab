"""Circuit-string DSL for linear lumped impedance models.

The syntax follows the convention popularised by ``impedance.py``::

    circuit := chain
    chain   := block ('-' block)*                    # series connection
    block   := element | 'p(' chain (',' chain)+ ')'  # parallel connection
    element := KIND INDEX

Element kinds and their impedances (``s`` is the Laplace variable, ``s = iω``
on the imaginary axis):

======  ===============================  =========================  ================
kind    element                          parameters                 Z(s)
======  ===============================  =========================  ================
``R``   resistor                         ``R{k}``                   R
``L``   ideal inductor                   ``L{k}``                   L s
``C``   ideal capacitor                  ``C{k}``                   1 / (C s)
``Q``   constant-phase element (CPE)     ``Q{k}``, ``alpha{k}``     1 / (Q s^α)
``Lf``  fractional inductor              ``Lf{k}``, ``beta{k}``     Lf s^β
======  ===============================  =========================  ================

The ``Q`` and ``Lf`` elements are the fractional-order generalisations of the
capacitor and the inductor (Westerlund & Ekstam 1994; Radwan & Salama 2011).
For ``alpha = beta = 1`` they reduce to ``C`` and ``L``.

The module turns a circuit string into

* an abstract syntax tree (``Element`` / ``Series`` / ``Parallel``),
* a SymPy expression ``Z(s)`` (series/parallel algebra),
* a netlist (the *circuit graph*: electrical nodes and branches), whose input
  impedance can be computed independently by nodal analysis.  The two routes
  must agree, which is checked in the test-suite.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

import sympy as sp

__all__ = [
    "Element",
    "Series",
    "Parallel",
    "KINDS",
    "parse",
    "to_string",
    "elements",
    "parameter_names",
    "parameter_symbol",
    "element_impedance",
    "impedance_expr",
    "netlist",
    "nodal_impedance",
]

# kind -> (parameter base names, short description)
KINDS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "R": (("R",), "resistor"),
    "L": (("L",), "inductor"),
    "C": (("C",), "capacitor"),
    "Q": (("Q", "alpha"), "constant-phase element (fractional capacitor)"),
    "Lf": (("Lf", "beta"), "fractional inductor"),
}

_TOKEN = re.compile(r"\s*(p\(|\(|\)|,|-|(?:Lf|R|L|C|Q)\d+)")
_ELEMENT = re.compile(r"^(Lf|R|L|C|Q)(\d+)$")


@dataclass(frozen=True)
class Element:
    """A single two-terminal element, e.g. ``Element('Q', '1')`` for ``Q1``."""

    kind: str
    index: str

    @property
    def name(self) -> str:
        return f"{self.kind}{self.index}"

    @property
    def params(self) -> Tuple[str, ...]:
        return tuple(f"{base}{self.index}" for base in KINDS[self.kind][0])


@dataclass(frozen=True)
class Series:
    children: Tuple["Node", ...]


@dataclass(frozen=True)
class Parallel:
    children: Tuple["Node", ...]


Node = Union[Element, Series, Parallel]


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    tokens, pos = [], 0
    text = text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None:
            raise ValueError(f"cannot parse circuit string at position {pos}: {text[pos:]!r}")
        tokens.append(m.group(1))
        pos = m.end()
    return tokens


def parse(text: str) -> Node:
    """Parse a circuit string into an AST.

    >>> to_string(parse("R0-p(R1,C1)"))
    'R0-p(R1,C1)'
    """
    tokens = _tokenize(text)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take(expected=None):
        nonlocal pos
        tok = peek()
        if tok is None or (expected is not None and tok != expected):
            raise ValueError(f"expected {expected!r}, got {tok!r} in {text!r}")
        pos += 1
        return tok

    def chain() -> Node:
        blocks = [block()]
        while peek() == "-":
            take("-")
            blocks.append(block())
        return blocks[0] if len(blocks) == 1 else Series(tuple(blocks))

    def block() -> Node:
        tok = peek()
        if tok == "p(":
            take("p(")
            branches = [chain()]
            while peek() == ",":
                take(",")
                branches.append(chain())
            take(")")
            if len(branches) < 2:
                raise ValueError("a parallel block needs at least two branches")
            return Parallel(tuple(branches))
        m = _ELEMENT.match(tok or "")
        if m is None:
            raise ValueError(f"unexpected token {tok!r} in {text!r}")
        take()
        return Element(m.group(1), m.group(2))

    tree = chain()
    if pos != len(tokens):
        raise ValueError(f"trailing tokens in {text!r}: {tokens[pos:]}")
    names = [e.name for e in elements(tree)]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate element names in {text!r}")
    return tree


def to_string(node: Node) -> str:
    """Inverse of :func:`parse` (canonical spelling)."""
    if isinstance(node, Element):
        return node.name
    if isinstance(node, Series):
        return "-".join(to_string(c) for c in node.children)
    return "p(" + ",".join(to_string(c) for c in node.children) + ")"


def elements(node: Node) -> List[Element]:
    """Elements in order of appearance (depth-first, left to right)."""
    if isinstance(node, Element):
        return [node]
    out: List[Element] = []
    for c in node.children:
        out.extend(elements(c))
    return out


def parameter_names(node: Node) -> List[str]:
    """Ordered parameter names, e.g. ``['R0', 'R1', 'Q1', 'alpha1']``."""
    out: List[str] = []
    for e in elements(node):
        out.extend(e.params)
    return out


# --------------------------------------------------------------------------
# symbolic impedance
# --------------------------------------------------------------------------
_PARAM = re.compile(r"^(Lf|R|L|C|Q|alpha|beta)(\d+)$")


def parameter_symbol(name: str) -> sp.Symbol:
    """SymPy symbol for a parameter name (``alpha1`` -> ``alpha_1`` -> α₁)."""
    m = _PARAM.match(name)
    if m is None:
        raise ValueError(f"not a circuit parameter name: {name!r}")
    base, idx = m.groups()
    base = {"Lf": "L^{f}"}.get(base, base)
    return sp.Symbol(f"{base}_{idx}", positive=True)


def element_impedance(e: Element, s: sp.Symbol, sym: Dict[str, sp.Symbol]) -> sp.Expr:
    p = [sym[n] for n in e.params]
    if e.kind == "R":
        return p[0]
    if e.kind == "L":
        return p[0] * s
    if e.kind == "C":
        return 1 / (p[0] * s)
    if e.kind == "Q":
        return 1 / (p[0] * s ** p[1])
    if e.kind == "Lf":
        return p[0] * s ** p[1]
    raise ValueError(e.kind)


def impedance_expr(node: Node, s: sp.Symbol, sym: Dict[str, sp.Symbol] | None = None) -> sp.Expr:
    """Impedance ``Z(s)`` by series/parallel algebra.

    Series: ``Z = Σ Z_k``; parallel: ``Z = 1 / Σ (1/Z_k)``.  A parallel block of
    two branches is written as ``Z_1 Z_2 / (Z_1 + Z_2)`` multiplied out, which
    for ``p(R, C)`` gives the familiar ``R / (1 + s R C)``.
    """
    if sym is None:
        sym = {n: parameter_symbol(n) for n in parameter_names(node)}
    if isinstance(node, Element):
        return element_impedance(node, s, sym)
    parts = [impedance_expr(c, s, sym) for c in node.children]
    if isinstance(node, Series):
        return sp.Add(*parts)
    # parallel: write via admittances, then clear the inner fractions
    y = sp.Add(*[1 / z for z in parts])
    return sp.simplify(1 / sp.together(y)) if len(parts) > 2 else _parallel2(parts[0], parts[1], s)


def _parallel2(z1: sp.Expr, z2: sp.Expr, s: sp.Symbol) -> sp.Expr:
    """Two-branch parallel impedance in the compact ``R / (1 + R Y)`` form.

    If one branch does not depend on ``s`` (a resistor), the result is written
    as ``R / (1 + R / Z_2)``, e.g. ``R / (1 + R Q s^α)`` for ``p(R, Q)``.
    """
    for r, other in ((z1, z2), (z2, z1)):
        if s not in r.free_symbols:
            return r / (1 + sp.powsimp(sp.expand(r / other)))
    return sp.cancel(z1 * z2 / (z1 + z2))


# --------------------------------------------------------------------------
# netlist (circuit graph) and nodal analysis
# --------------------------------------------------------------------------
def netlist(node: Node) -> List[Tuple[Element, int, int]]:
    """Branch list ``(element, n_from, n_to)`` of the circuit graph.

    The measurement port is between node ``1`` (terminal) and node ``0``
    (ground).  Series connections create internal nodes ``2, 3, ...``.
    """
    branches: List[Tuple[Element, int, int]] = []
    counter = [2]

    def build(n: Node, a: int, b: int) -> None:
        if isinstance(n, Element):
            branches.append((n, a, b))
        elif isinstance(n, Parallel):
            for c in n.children:
                build(c, a, b)
        else:  # series
            nodes = [a] + [None] * (len(n.children) - 1) + [b]
            for i in range(1, len(n.children)):
                nodes[i] = counter[0]
                counter[0] += 1
            for i, c in enumerate(n.children):
                build(c, nodes[i], nodes[i + 1])

    build(node, 1, 0)
    return branches


def nodal_impedance(branches: Sequence[Tuple[Element, int, int]], s: sp.Symbol,
                    sym: Dict[str, sp.Symbol] | None = None) -> sp.Expr:
    """Input impedance between nodes 1 and 0 by nodal analysis.

    With the reduced incidence matrix ``A`` (ground row removed) and the
    branch admittances ``y_b = 1 / Z_b`` the node equations read
    ``(A diag(y) Aᵀ) v = j``.  Injecting a unit current into node 1 gives
    ``Z_in = v_1``.  This route never uses the series/parallel structure, so it
    is an independent check of :func:`impedance_expr`.
    """
    if sym is None:
        names: List[str] = []
        for e, _, _ in branches:
            names.extend(e.params)
        sym = {n: parameter_symbol(n) for n in names}
    nodes = sorted({n for _, a, b in branches for n in (a, b)} - {0})
    index = {n: i for i, n in enumerate(nodes)}
    A = sp.zeros(len(nodes), len(branches))
    for k, (_, a, b) in enumerate(branches):
        if a != 0:
            A[index[a], k] = 1
        if b != 0:
            A[index[b], k] = -1
    y = sp.diag(*[1 / element_impedance(e, s, sym) for e, _, _ in branches])
    Y = A * y * A.T
    j = sp.zeros(len(nodes), 1)
    j[index[1], 0] = 1
    v = Y.LUsolve(j)
    return sp.simplify(v[index[1], 0])
