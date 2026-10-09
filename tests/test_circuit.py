"""Circuit DSL: parsing, symbolic impedance, nodal analysis, passivity."""
import numpy as np
import pytest
import sympy as sp

from rlcfreq import circuit as c

CIRCUITS = ["R0-L0-C0", "R0-Lf0-Q0", "R0-L0-p(R1,C1)", "L0-R0-p(R1,C1)-p(R2,C2)-C3",
            "L0-R0-p(R1,Q1)-p(R2,Q2)-Q3", "p(R0,C0,L0)", "p(R0-C0,L0)-R1"]


@pytest.mark.parametrize("text", CIRCUITS)
def test_parse_roundtrip(text):
    assert c.to_string(c.parse(text)) == text


@pytest.mark.parametrize("bad", ["R0-", "p(R0)", "R0-X1", "R0-R0", "p(R0,C0", ""])
def test_parse_errors(bad):
    with pytest.raises((ValueError, IndexError)):
        c.parse(bad)


def test_parameter_names_order():
    assert c.parameter_names(c.parse("L0-R0-p(R1,Q1)")) == ["L0", "R0", "R1", "Q1", "alpha1"]


def test_textbook_impedance_is_classic():
    s = sp.Symbol("s")
    Z = c.impedance_expr(c.parse("R0-L0-C0"), s)
    R, L, C = (c.parameter_symbol(n) for n in ("R0", "L0", "C0"))
    assert sp.simplify(Z - (R + L * s + 1 / (C * s))) == 0


def test_parallel_rc_compact_form():
    s = sp.Symbol("s")
    Z = c.impedance_expr(c.parse("p(R1,Q1)"), s)
    R, Q, a = (c.parameter_symbol(n) for n in ("R1", "Q1", "alpha1"))
    assert sp.simplify(Z - R / (1 + R * Q * s ** a)) == 0


@pytest.mark.parametrize("text", CIRCUITS)
def test_nodal_analysis_matches_series_parallel_algebra(text):
    """Two independent routes (graph + node equations vs. algebra) agree."""
    tree = c.parse(text)
    s = sp.Symbol("s")
    Z1 = c.impedance_expr(tree, s)
    Z2 = c.nodal_impedance(c.netlist(tree), s)
    rng = np.random.default_rng(0)
    syms = sorted(Z1.free_symbols | Z2.free_symbols, key=str)
    for _ in range(3):
        vals = {x: (0.3 + 1.2j if x == s else rng.uniform(0.2, 1.0)) for x in syms}
        assert abs(complex(Z1.subs(vals)) - complex(Z2.subs(vals))) < 1e-10


def test_netlist_is_a_connected_graph_with_port():
    br = c.netlist(c.parse("L0-R0-p(R1,Q1)-p(R2,Q2)-Q3"))
    nodes = {n for _, a, b in br for n in (a, b)}
    assert {0, 1} <= nodes and len(br) == 7
    # parallel elements share both terminals
    pairs = {e.name: (a, b) for e, a, b in br}
    assert pairs["R1"] == pairs["Q1"] and pairs["R2"] == pairs["Q2"]


@pytest.mark.parametrize("text", ["R0-Lf0-Q0", "L0-R0-p(R1,Q1)-p(R2,Q2)-Q3", "R0-L0-p(R1,Q1)"])
def test_passivity_re_z_nonnegative(text):
    """Proposition (passivity): for α, β ∈ (0, 1] and positive elements Re Z(iω) ≥ 0."""
    tree = c.parse(text)
    s = sp.Symbol("s")
    names = c.parameter_names(tree)
    syms = [c.parameter_symbol(n) for n in names]
    f = sp.lambdify(syms + [s], c.impedance_expr(tree, s), "numpy")
    rng = np.random.default_rng(1)
    w = np.logspace(-4, 7, 400)
    for _ in range(50):
        vals = [rng.uniform(0.05, 1.0) if n.startswith(("alpha", "beta")) else 10 ** rng.uniform(-6, 3)
                for n in names]
        Z = f(*vals, 1j * w)
        assert np.all(Z.real >= -1e-12 * np.abs(Z))
