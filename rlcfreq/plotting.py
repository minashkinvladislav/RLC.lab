"""Plotting helpers (matplotlib): style, Nyquist/Bode, graphs, 3-D views.

Colours follow a colour-blind-validated categorical order (blue, orange,
aqua, yellow, magenta, green, violet, red) and a single-hue blue ramp for
magnitudes such as cycle number.  Grids are hairlines; one y-axis per panel.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .graph import ExpressionDAG, layered_layout

__all__ = ["PALETTE", "KIND_COLORS", "BLUES", "apply_style", "nyquist", "bode", "draw_dag",
           "draw_adjacency", "draw_circuit_graph", "savefig"]

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
KIND_COLORS = {"observed": "#4C9AFF", "parameter": "#FFAB4C", "constant": "#d9d8d4",
               "operation": "#8ED18E", "output": "#ffffff"}
BLUES = LinearSegmentedColormap.from_list(
    "blues_seq", ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6,
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.prop_cycle": mpl.cycler(color=PALETTE), "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.titleweight": "regular", "lines.linewidth": 1.6, "lines.markersize": 4.5,
        "legend.frameon": False, "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
    })


def nyquist(ax, Z, fmt="o", label=None, **kw):
    """Nyquist plot ``(Z', −Z'')`` with equal axis scaling."""
    ax.plot(np.real(Z), -np.imag(Z), fmt, label=label, **kw)
    ax.set_xlabel("Z′, Ом")
    ax.set_ylabel("−Z″, Ом")
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def bode(axes, f, Z, fmt="o", label=None, **kw):
    a1, a2 = axes
    a1.loglog(f, np.abs(Z), fmt, label=label, **kw)
    a2.semilogx(f, np.degrees(np.angle(Z)), fmt, label=label, **kw)
    a1.set_ylabel("|Z|, Ом")
    a2.set_ylabel("arg Z, °")
    a2.set_xlabel("f, Гц")
    return axes


def draw_dag(dag: ExpressionDAG, ax=None, title: str = "", node_size: int = 900, font_size: int = 8,
             highlight_shared: bool = True):
    """Draw an :class:`ExpressionDAG` left-to-right (leaves → outputs)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    G = dag.to_networkx()
    pos = layered_layout(dag, x_gap=1.6, y_gap=0.9)
    colors = [KIND_COLORS[G.nodes[n]["kind"]] for n in G.nodes]
    shared = {n.id for n in dag.shared_nodes()} if highlight_shared else set()
    edgec = ["#d2386c" if n in shared else "#3a3a38" for n in G.nodes]
    lw = [2.6 if n in shared else 0.8 for n in G.nodes]
    outs = [n.id for n in dag.outputs]
    inner = [n for n in G.nodes if n not in outs]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#6f6e69", width=0.9, arrows=True, arrowsize=10,
                           node_size=node_size, connectionstyle="arc3,rad=0.04")
    nx.draw_networkx_nodes(G, pos, nodelist=inner, ax=ax, node_color=[colors[i] for i in inner],
                           edgecolors=[edgec[i] for i in inner], linewidths=[lw[i] for i in inner],
                           node_size=node_size)
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["label"] for n in inner}, ax=ax,
                            font_size=font_size)
    for o in outs:
        x, y = pos[o]
        ax.text(x - 0.12, y, G.nodes[o]["label"], fontsize=font_size + 2, fontweight="bold",
                va="center", ha="left")
    ax.set_title(title)
    ax.axis("off")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=9, mfc=KIND_COLORS[k], mec="#3a3a38", label=lab)
               for k, lab in (("observed", "наблюдаемая переменная X"), ("parameter", "параметр θ"),
                              ("constant", "константа"), ("operation", "операция V"))]
    if shared:
        handles.append(Line2D([], [], marker="o", ls="", ms=9, mfc="white", mec="#d2386c", mew=2.4,
                              label="общий (shared) узел"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    return ax


def draw_adjacency(dag: ExpressionDAG, ax=None, title: str = "Матрица связей Z"):
    """Heat-map of the binary link matrix (``Z[i, j] = 1`` if j is an input of i)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    Z = (dag.adjacency() > 0).astype(int)
    labels = [n.label.replace("\n", " ") for n in dag.nodes]
    ax.imshow(Z, cmap=LinearSegmentedColormap.from_list("bin", ["#ffffff", "#104281"]), aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("j (вход)")
    ax.set_ylabel("i (узел)")
    ax.grid(False)
    ax.set_title(title)
    return ax


def draw_circuit_graph(G: nx.MultiGraph, ax=None, title: str = ""):
    """Circuit graph: electrical nodes on a line, element branches as edges.

    Parallel elements are drawn as quadratic Bézier arcs (computed in data
    coordinates so that labels sit on the curves).  Node ``0`` is ground (⊥).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    nodes = sorted(G.nodes, key=lambda n: (n == 0, n))
    pos = {n: np.array([float(i), 0.0]) for i, n in enumerate(nodes)}
    seen: Dict[tuple, int] = {}
    ymin, ymax = -0.2, 0.2
    for a, b, k, d in G.edges(keys=True, data=True):
        key = tuple(sorted((a, b)))
        m = seen.get(key, 0)
        seen[key] = m + 1
        A, B = pos[a], pos[b]
        if A[0] > B[0]:
            A, B = B, A
        span = B[0] - A[0]
        h = 0.0 if (m == 0 and span < 1.5) else (0.45 * (1 + m // 2) * (1 if m % 2 else -1) if span < 1.5
                                                 else 0.55 * (1 + m))
        ctrl = (A + B) / 2 + np.array([0.0, 2 * h])          # Bézier control point
        tt = np.linspace(0, 1, 60)[:, None]
        P = (1 - tt) ** 2 * A + 2 * (1 - tt) * tt * ctrl + tt ** 2 * B
        color = PALETTE[["R", "L", "C", "Q", "Lf"].index(d["kind"])]
        ax.plot(P[:, 0], P[:, 1], color=color, lw=2.4, zorder=1, solid_capstyle="round")
        mid = P[30]
        ax.text(mid[0], mid[1] + (0.07 if h >= 0 else -0.07), d["element"], ha="center",
                va="bottom" if h >= 0 else "top", fontsize=9, color="#0b0b0b")
        ymin, ymax = min(ymin, P[:, 1].min() - 0.25), max(ymax, P[:, 1].max() + 0.25)
    for n, p in pos.items():
        ax.scatter([p[0]], [p[1]], s=420, zorder=2, color="#d9d8d4" if n in (0, 1) else "#ffffff",
                   edgecolors="#3a3a38")
        ax.text(p[0], p[1], "⊥" if n == 0 else str(n), ha="center", va="center", fontsize=10, zorder=3)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.4, len(nodes) - 0.6)
    ax.set_title(title)
    ax.axis("off")
    return ax


def savefig(fig, path, **kw):
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", **kw)
