"""Expression DAG: dict / link matrix / networkx encodings round-trip exactly."""
import json

import networkx as nx
import numpy as np
import sympy as sp

from rlcfreq import symbolic as S
from rlcfreq.graph import ExpressionDAG, circuit_graph
from rlcfreq import circuit as c


def _fractional_dag():
    re_, im_ = S.split_real_imag(S.fractional_series_rlc()["Z"])
    params = {S.R: 10.0, S.L: 1e-3, S.Q: 1e-6, S.alpha: 0.9, S.beta: 1.0}
    return ExpressionDAG.from_sympy({"Re Z": re_, "Im Z": im_}, observed=[S.omega], parameters=params), (re_, im_)


def test_topological_order_and_kinds():
    dag, _ = _fractional_dag()
    kinds = [n.kind for n in dag.nodes]
    assert kinds[0] == "observed" and kinds[-1] == "output"
    assert nx.is_directed_acyclic_graph(dag.to_networkx())


def test_shared_subtrees_exist_for_two_outputs():
    dag, _ = _fractional_dag()
    shared = dag.shared_nodes()
    assert len(shared) >= 2           # e.g. ω^β·L and 1/(Q ω^α) feed Re and Im


def test_dict_roundtrip_and_reconstruction():
    dag, (re_, im_) = _fractional_dag()
    d = json.loads(dag.to_json())
    dag2 = ExpressionDAG.from_dict(d)
    out = dag2.to_sympy()
    assert sp.simplify(out["Re Z"] - re_) == 0
    assert sp.simplify(out["Im Z"] - im_) == 0


def test_link_matrix_roundtrip():
    dag, _ = _fractional_dag()
    Z = dag.adjacency(positions=True)
    assert np.all(np.triu(Z) == 0)    # strictly lower-triangular: inputs precede nodes
    dag2 = ExpressionDAG.from_adjacency(Z, dag.metadata())
    assert [n.args for n in dag2.nodes] == [n.args for n in dag.nodes]


def test_graph_interpreter_matches_sympy():
    dag, (re_, im_) = _fractional_dag()
    w = np.logspace(1, 6, 7)
    out = dag.evaluate({"omega": w})
    vals = {S.R: 10.0, S.L: 1e-3, S.Q: 1e-6, S.alpha: 0.9, S.beta: 1.0}
    f_re = sp.lambdify(S.omega, re_.subs(vals), "numpy")
    f_im = sp.lambdify(S.omega, im_.subs(vals), "numpy")
    assert np.allclose(out["Re Z"], f_re(w)) and np.allclose(out["Im Z"], f_im(w))


def test_state_space_dag_shares_parameter_leaves():
    ss = S.state_space()
    q, i, v = ss["symbols"]
    dag = ExpressionDAG.from_sympy({"dq/dt": ss["dq/dt"], "di/dt": ss["di/dt"]}, observed=[q, i, v])
    cnt = dag.counts()
    assert cnt["observed"] == 3 and cnt["parameter"] == 3 and cnt["output"] == 2


def test_circuit_graph_cycle_rank():
    """A parallel block adds an independent loop: cycle rank = E − V + 1."""
    G = circuit_graph(c.parse("L0-R0-p(R1,Q1)-p(R2,Q2)-Q3"))
    E, V = G.number_of_edges(), G.number_of_nodes()
    assert E - V + 1 == 2
