"""rlcfreq – RLC circuits as ODE models in the frequency domain.

Modules
-------
circuit     circuit-string DSL, symbolic impedance, netlist and nodal analysis
models      parameter vectors and compiled models (impedance, resonator S21)
symbolic    ODE -> Laplace -> Z(s), fractional variant, resonance, transfer functions
graph       expression DAG (dict / link matrix / networkx), circuit graph
fields      vector fields R^3 -> R^3: div, curl, Jacobian, products, Liouville
fitting     error function, optimisers (TRF/LM, DE, CMA-ES, multistart), covariance
timedomain  ODE solvers, Grünwald–Letnikov scheme, Talbot inverse Laplace
data        HTTP download with SHA-256, parsers (ZPlot, Touchstone), loaders
plotting    matplotlib helpers
"""
from . import circuit, data, fields, fitting, graph, models, symbolic, timedomain  # noqa: F401

__version__ = "1.0.0"
