#!/usr/bin/env python3
"""Execute a Jupyter notebook without Jupyter and store the outputs in place.

Normally use ``jupyter nbconvert --to notebook --execute --inplace``.  This
fallback needs only the standard library and matplotlib; it reproduces what
the IPython kernel would store for this notebook: printed text, the value of
the last expression of a cell (plain text, HTML, LaTeX), ``display(...)``
calls and matplotlib figures (PNG).  Cells run in one shared namespace, in
order; execution stops at the first error (unless ``--allow-errors``).

Usage::

    python tools/execute_notebook.py notebooks/lab1_rlc_frequency.ipynb [--allow-errors]
"""
from __future__ import annotations

import ast
import base64
import contextlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _lines(text: str):
    parts = text.splitlines(keepends=True)
    return parts


def _bundle(obj):
    """MIME bundle for an object, mimicking IPython's formatters."""
    data = {"text/plain": _lines(repr(obj))}
    if hasattr(obj, "_repr_html_"):
        try:
            html = obj._repr_html_()
            if html:
                data["text/html"] = _lines(html)
        except Exception:
            pass
    if hasattr(obj, "_repr_latex_"):
        try:
            latex = obj._repr_latex_()
            if latex:
                data["text/latex"] = _lines(latex)
        except Exception:
            pass
    if isinstance(obj, matplotlib.figure.Figure):
        data = {"image/png": _png(obj), "text/plain": [repr(obj)]}
    return data


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=fig.dpi)
    return base64.b64encode(buf.getvalue()).decode()


class Runner:
    def __init__(self, cwd: Path):
        self.ns = {"__name__": "__main__", "display": self.display}
        self.outputs = []
        self.cwd = cwd
        plt.switch_backend("Agg")      # initialise the backend before patching pyplot

        def show(*args, **kwargs):     # figures are emitted where plt.show() is called
            self._show()

        plt.show = show

    def display(self, *objs, **kw):
        self._flush_stream()
        for obj in objs:
            self.outputs.append({"output_type": "display_data", "data": _bundle(obj), "metadata": {}})

    def _show(self, *a, **kw):
        self._flush_stream()
        for num in plt.get_fignums():
            fig = plt.figure(num)
            self.outputs.append({"output_type": "display_data",
                                 "data": {"image/png": _png(fig), "text/plain": [repr(fig)]},
                                 "metadata": {}})
        plt.close("all")

    def _flush_stream(self):
        for name, buf in (("stdout", self.out), ("stderr", self.err)):
            text = buf.getvalue()
            if text:
                self.outputs.append({"output_type": "stream", "name": name, "text": _lines(text)})
                buf.seek(0)
                buf.truncate()

    def run_cell(self, source: str, count: int):
        self.outputs = []
        self.out, self.err = io.StringIO(), io.StringIO()
        error = None
        tree = ast.parse(source)
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        with contextlib.redirect_stdout(self.out), contextlib.redirect_stderr(self.err):
            try:
                exec(compile(tree, f"<cell {count}>", "exec"), self.ns)
                if last is not None:
                    value = eval(compile(last, f"<cell {count}>", "eval"), self.ns)
                    if value is not None and not (
                            isinstance(value, (list, tuple)) and value and
                            all(isinstance(v, matplotlib.artist.Artist) for v in value)) and \
                            not isinstance(value, matplotlib.artist.Artist):
                        self._flush_stream()
                        self.outputs.append({"output_type": "execute_result", "execution_count": count,
                                             "data": _bundle(value), "metadata": {}})
            except Exception as exc:
                error = exc
                tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        self._flush_stream()
        self._show()
        if error is not None:
            self.outputs.append({"output_type": "error", "ename": type(error).__name__,
                                 "evalue": str(error), "traceback": tb})
        return self.outputs, error


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("notebook")
    p.add_argument("--allow-errors", action="store_true")
    a = p.parse_args(argv)
    path = Path(a.notebook).resolve()
    nb = json.loads(path.read_text(encoding="utf8"))
    os.chdir(path.parent)  # like Jupyter: the kernel starts in the notebook folder
    sys.path.insert(0, str(path.parent))
    runner = Runner(path.parent)
    count = 0
    t_all = time.perf_counter()
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        count += 1
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        t0 = time.perf_counter()
        outs, err = runner.run_cell(src, count)
        cell["outputs"] = outs
        cell["execution_count"] = count
        print(f"[{count:3d}] {time.perf_counter() - t0:7.2f} s  {src.strip().splitlines()[0][:70] if src.strip() else ''}",
              flush=True)
        if err is not None:
            print("".join(outs[-1]["traceback"]), file=sys.stderr)
            if not a.allow_errors:
                path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
                sys.exit(1)
    nb.setdefault("metadata", {})["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                                   "name": "python3"}
    nb["metadata"]["language_info"] = {"name": "python", "version": sys.version.split()[0]}
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    print(f"done: {count} cells in {time.perf_counter() - t_all:.1f} s -> {path}")


if __name__ == "__main__":
    main()
