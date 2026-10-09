"""Graph representations of symbolic models.

:class:`ExpressionDAG` turns one or several SymPy expressions (e.g. the
right-hand side of an ODE system, or the real and imaginary parts of
``Z(iω)``) into a directed acyclic graph of a *superposition* of primitive
functions, following the notation of the course (Strijov):

* leaves – observed variables ``X`` (blue), parameters ``θ`` (orange) and
  numeric constants (grey);
* inner nodes – primitive operations ``V`` from the set ``G`` (green);
* output nodes – the modelled quantities (``dq/dt``, ``Re Z`` ...).

Identical sub-expressions are stored once (hash-consing), so a subtree used by
several outputs becomes a *shared node* (out-degree ≥ 2).

Three interchangeable encodings are provided and round-trip exactly:

``to_dict()``        node id -> {kind, label, op, args, ...}  (JSON-friendly)
``adjacency()``      ``Z[i, j] = p`` if node ``j`` is the ``p``-th argument of
                     node ``i`` (``p ≥ 1``); ``Z > 0`` is the binary link matrix.
                     Keeping the argument position makes non-commutative
                     operations (powers) reconstructible from the matrix alone.
``to_networkx()``    ``networkx.DiGraph`` with edges ``j → i``.

:func:`circuit_graph` builds the *other* graph of an electrical model – the
circuit itself (electrical nodes, element branches) – as a networkx
``MultiGraph``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import networkx as nx
import numpy as np
import sympy as sp

from . import circuit as cdsl

__all__ = ["DagNode", "ExpressionDAG", "circuit_graph", "layered_layout"]

OP_LABELS = {"Add": "+", "Mul": "×", "Pow": "^", "inv": "1/·", "neg": "−·", "sqrt": "√", "sin": "sin",
             "cos": "cos", "exp": "exp", "log": "ln", "tan": "tan", "cot": "cot", "re": "Re",
             "im": "Im", "Abs": "|·|", "output": "="}


@dataclass
class DagNode:
    id: int
    kind: str                      # observed | parameter | constant | operation | output
    label: str
    op: Optional[str] = None
    args: List[int] = field(default_factory=list)
    name: Optional[str] = None     # symbol name (leaves) / output name
    value: Optional[float] = None  # parameter value or numeric constant
    srepr: Optional[str] = None    # exact constant as text ("-1", "1/2", "pi", "I")


class ExpressionDAG:
    """DAG of a superposition built from SymPy expressions."""

    def __init__(self, nodes: List[DagNode]):
        self.nodes = nodes
        self._check()

    # ------------------------------------------------------------------
    @classmethod
    def from_sympy(cls, outputs: Mapping[str, sp.Expr], observed: Iterable[sp.Symbol],
                   parameters: Optional[Mapping[sp.Symbol, Optional[float]]] = None,
                   labels: Optional[Mapping[sp.Symbol, str]] = None) -> "ExpressionDAG":
        observed = set(observed)
        parameters = dict(parameters or {})
        labels = dict(labels or {})
        memo: Dict[sp.Basic, dict] = {}
        order: List[dict] = []

        def leaf_kind(e):
            if isinstance(e, sp.Symbol):
                return "observed" if e in observed else "parameter"
            return "constant"

        def visit(e: sp.Basic) -> dict:
            if e in memo:
                return memo[e]
            if e.is_Atom:
                kind = leaf_kind(e)
                if kind == "constant":
                    lab = {"I": "i", "pi": "π", "E": "e"}.get(str(e), sp.sstr(e))
                    try:
                        val = float(e)
                    except TypeError:
                        val = None
                    rec = dict(kind=kind, label=lab, srepr=_encode_constant(e), value=val)
                else:
                    lab = labels.get(e, sp.sstr(e))
                    val = parameters.get(e)
                    if kind == "parameter" and val is not None:
                        lab = f"{lab}\n={val:.3g}"
                    rec = dict(kind=kind, label=lab, name=e.name, value=val)
                rec["expr"] = e
                rec["args"] = []
                memo[e] = rec
                order.append(rec)
                return rec
            # operations with readable special cases
            if isinstance(e, sp.Pow) and e.exp == -1:
                op, kids = "inv", [e.base]
            elif isinstance(e, sp.Pow) and e.exp == sp.Rational(1, 2):
                op, kids = "sqrt", [e.base]
            elif isinstance(e, sp.Mul) and len(e.args) == 2 and e.args[0] == -1:
                op, kids = "neg", [e.args[1]]
            else:
                op, kids = type(e).__name__, list(e.args)
            child_recs = [visit(k) for k in kids]
            rec = dict(kind="operation", label=OP_LABELS.get(op, op), op=op, expr=e,
                       args_rec=child_recs, args=[])
            memo[e] = rec
            order.append(rec)
            return rec

        out_recs = []
        for name, expr in outputs.items():
            root = visit(sp.sympify(expr))
            out_recs.append(dict(kind="output", label=f"= {name}", op="output", name=name,
                                 args_rec=[root], args=[], expr=None))

        # ids: leaves (observed, parameters, constants), then operations in
        # creation order (children are created before parents), then outputs
        rank = {"observed": 0, "parameter": 1, "constant": 2}
        leaves = sorted([r for r in order if r["kind"] != "operation"],
                        key=lambda r: (rank[r["kind"]], r["label"]))
        ops = [r for r in order if r["kind"] == "operation"]
        allrecs = leaves + ops + out_recs
        for i, r in enumerate(allrecs):
            r["id"] = i
        nodes = []
        for r in allrecs:
            args = [c["id"] for c in r.get("args_rec", [])]
            nodes.append(DagNode(r["id"], r["kind"], r["label"], r.get("op"), args, r.get("name"),
                                 r.get("value"), r.get("srepr")))
        dag = cls(nodes)
        dag._exprs = {r["id"]: r["expr"] for r in allrecs if r.get("expr") is not None}
        return dag

    def _check(self) -> None:
        for n in self.nodes:
            for a in n.args:
                if not (0 <= a < n.id):
                    raise ValueError(f"node {n.id}: argument {a} violates topological order")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def outputs(self) -> List[DagNode]:
        return [n for n in self.nodes if n.kind == "output"]

    def out_degree(self) -> np.ndarray:
        return (self.adjacency() > 0).sum(axis=0)

    def shared_nodes(self) -> List[DagNode]:
        """Operation nodes used by more than one parent (shared subtrees)."""
        deg = self.out_degree()
        return [n for n in self.nodes if n.kind == "operation" and deg[n.id] >= 2]

    def counts(self) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for n in self.nodes:
            c[n.kind] = c.get(n.kind, 0) + 1
        c["edges"] = int((self.adjacency() > 0).sum())
        c["shared"] = len(self.shared_nodes())
        return c

    # ---- encodings -----------------------------------------------------
    def to_dict(self) -> Dict[int, dict]:
        out = {}
        for n in self.nodes:
            d = dict(kind=n.kind, label=n.label, op=n.op, args=list(n.args))
            if n.name is not None:
                d["name"] = n.name
            if n.value is not None:
                d["value"] = n.value
            if n.srepr is not None:
                d["srepr"] = n.srepr
            out[n.id] = d
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=1)

    @classmethod
    def from_dict(cls, d: Mapping) -> "ExpressionDAG":
        nodes = []
        for k in sorted(d, key=int):
            v = d[k]
            nodes.append(DagNode(int(k), v["kind"], v["label"], v.get("op"), list(v.get("args", [])),
                                 v.get("name"), v.get("value"), v.get("srepr")))
        return cls(nodes)

    def adjacency(self, positions: bool = True) -> np.ndarray:
        """Link matrix ``Z``: ``Z[i, j] = p`` if ``j`` is the p-th argument of ``i``."""
        n = len(self.nodes)
        Z = np.zeros((n, n), dtype=int)
        for node in self.nodes:
            for p, a in enumerate(node.args, start=1):
                Z[node.id, a] = p if positions else 1
        return Z

    @classmethod
    def from_adjacency(cls, Z: np.ndarray, meta: Sequence[Mapping]) -> "ExpressionDAG":
        """Rebuild from the positional link matrix and per-node metadata
        (``kind``, ``label``, ``op``, ``name``, ``srepr``) – no ``args`` needed."""
        nodes = []
        for i, m in enumerate(meta):
            row = Z[i]
            js = np.nonzero(row)[0]
            args = [int(j) for j in sorted(js, key=lambda j: row[j])]
            nodes.append(DagNode(i, m["kind"], m["label"], m.get("op"), args, m.get("name"),
                                 m.get("value"), m.get("srepr")))
        return cls(nodes)

    def metadata(self) -> List[dict]:
        return [{k: v for k, v in self.to_dict()[n.id].items() if k != "args"} for n in self.nodes]

    def to_networkx(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for n in self.nodes:
            G.add_node(n.id, kind=n.kind, label=n.label, op=n.op)
        for n in self.nodes:
            for p, a in enumerate(n.args, start=1):
                G.add_edge(a, n.id, position=p)
        return G

    # ---- back to SymPy -------------------------------------------------
    def to_sympy(self, assumptions: Optional[Mapping[str, dict]] = None) -> Dict[str, sp.Expr]:
        """Reconstruct the output expressions from the graph alone."""
        assumptions = assumptions or {}
        val: Dict[int, sp.Expr] = {}
        outs: Dict[str, sp.Expr] = {}
        for n in self.nodes:
            if n.kind in {"observed", "parameter"}:
                val[n.id] = sp.Symbol(n.name, **assumptions.get(n.name, {"positive": True}))
            elif n.kind == "constant":
                val[n.id] = _decode_constant(n.srepr) if n.srepr else sp.Float(n.value)
            elif n.kind == "operation":
                a = [val[i] for i in n.args]
                if n.op == "inv":
                    val[n.id] = 1 / a[0]
                elif n.op == "neg":
                    val[n.id] = -a[0]
                elif n.op == "sqrt":
                    val[n.id] = sp.sqrt(a[0])
                else:
                    val[n.id] = getattr(sp, n.op)(*a)
            else:  # output
                outs[n.name] = val[n.args[0]]
        return outs

    # ---- numeric evaluation (graph interpreter) ------------------------
    def evaluate(self, values: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Evaluate the outputs by walking the graph in topological order."""
        fns = {"Add": lambda *a: sum(a[1:], a[0]), "Mul": lambda *a: _prod(a), "Pow": lambda a, b: a ** b,
               "inv": lambda a: 1 / a, "neg": lambda a: -a, "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos,
               "exp": np.exp, "log": np.log, "tan": np.tan, "cot": lambda a: 1 / np.tan(a)}
        val: Dict[int, np.ndarray] = {}
        outs = {}
        for n in self.nodes:
            if n.kind in {"observed", "parameter"}:
                val[n.id] = np.asarray(values[n.name] if n.name in values else n.value)
            elif n.kind == "constant":
                val[n.id] = complex(_decode_constant(n.srepr)) if n.srepr else n.value
                if np.imag(val[n.id]) == 0:
                    val[n.id] = float(np.real(val[n.id]))
            elif n.kind == "operation":
                val[n.id] = fns[n.op](*[val[i] for i in n.args])
            else:
                outs[n.name] = val[n.args[0]]
        return outs


_NAMED_CONSTANTS = {"I": sp.I, "pi": sp.pi, "E": sp.E}


def _encode_constant(e: sp.Basic) -> str:
    """Exact text form of a numeric atom: ``-1``, ``1/2``, ``0.25``, ``pi``, ``I``."""
    if e in _NAMED_CONSTANTS.values():
        return str(e)
    if e.is_Rational:
        return f"{e.p}/{e.q}" if e.q != 1 else str(e.p)
    if e.is_Float:
        return repr(float(e))
    raise ValueError(f"unsupported constant {e!r}")


def _decode_constant(text: str) -> sp.Expr:
    """Inverse of :func:`_encode_constant` (no ``eval``: safe for untrusted JSON)."""
    if text in _NAMED_CONSTANTS:
        return _NAMED_CONSTANTS[text]
    if "/" in text:
        p, q = text.split("/")
        return sp.Rational(int(p), int(q))
    try:
        return sp.Integer(int(text))
    except ValueError:
        return sp.Float(float(text))


def _prod(a):
    out = a[0]
    for x in a[1:]:
        out = out * x
    return out


def layered_layout(dag: ExpressionDAG, x_gap: float = 1.0, y_gap: float = 1.0) -> Dict[int, tuple]:
    """Left-to-right layered layout: column = longest path from the leaves."""
    depth: Dict[int, int] = {}
    for n in dag.nodes:
        depth[n.id] = 0 if not n.args else 1 + max(depth[a] for a in n.args)
    maxd = max(depth.values())
    for n in dag.outputs:
        depth[n.id] = maxd  # align outputs in the last column
    cols: Dict[int, List[int]] = {}
    for n in dag.nodes:
        cols.setdefault(depth[n.id], []).append(n.id)
    pos = {}
    # order nodes in a column by the mean height of their arguments (fewer crossings)
    for d in sorted(cols):
        ids = cols[d]
        if d > 0:
            ids.sort(key=lambda i: np.mean([pos[a][1] for a in dag.nodes[i].args]) if dag.nodes[i].args else 0)
        m = len(ids)
        for k, i in enumerate(ids):
            pos[i] = (d * x_gap, (k - (m - 1) / 2) * y_gap)
    return pos


def circuit_graph(tree: cdsl.Node) -> nx.MultiGraph:
    """Circuit graph: electrical nodes as vertices, elements as (multi-)edges."""
    G = nx.MultiGraph()
    for e, a, b in cdsl.netlist(tree):
        G.add_edge(a, b, key=e.name, element=e.name, kind=e.kind)
    G.graph["port"] = (1, 0)
    return G
