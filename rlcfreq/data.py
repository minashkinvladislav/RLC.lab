"""Data sources: HTTP download with checksums, parsers and loaders.

All files are fetched over HTTPS from **pinned commits** of public GitHub
repositories and verified with SHA-256, so the analysis is reproducible
byte-for-byte.  Files are cached in ``data/raw`` (or ``$RLCFREQ_DATA_DIR``);
a cached file is used only if its checksum matches.

Sources
-------
``zhang2020``
    Y. Zhang, Q. Tang, Y. Zhang, J. Wang, U. Stimming, A. A. Lee.
    *Identifying degradation patterns of lithium ion batteries from impedance
    spectroscopy using machine learning.* Nat. Commun. 11, 1706 (2020).
    Code repository of the paper (``Code-Matlab.zip``): EIS spectra of the
    45 mAh LiCoO₂/graphite coin cell **35C02** (35 °C) at state V (15 min rest
    after full charge), every second cycle 2…598, 60 frequencies 20 kHz…0.02 Hz,
    plus the discharge capacity.  Raw data: Zenodo, doi:10.5281/zenodo.3633835.
``impedancepy``
    M. D. Murbach, B. Gerwe, N. Dawson-Elli, L. Tsui. *impedance.py: A Python
    package for electrochemical impedance analysis.* JOSS 5(52), 2349 (2020).
    ``data/Circuit{1,2,3}_EIS_{1,2}.z``: ZPlot measurements (10 mV, 1 Hz…
    50/300/150 kHz) of three physical test circuits ``R0-(R1‖C1)`` built from
    discrete components, each measured twice; ``data/exampleData.csv``: EIS
    of a Li-ion cell.
``skrf``
    A. Arsenovic et al. *scikit-rf: An Open Source Python Package for Microwave
    Network Creation, Analysis, and Calibration.* IEEE Microwave Magazine
    23(1), 98–105 (2022).  ``doc/source/examples/qfactor/data``: two-port
    S-parameters of 36, 72 and 144 mm stripline resonators (PCB laminate
    characterisation), Keysight N5242A VNA, 1–5 GHz.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "SOURCES",
    "data_dir",
    "fetch",
    "fetch_all",
    "sha256",
    "parse_zplot",
    "parse_touchstone",
    "zhang_frequency_grid",
    "EISSpectrum",
    "BatterySeries",
    "TwoPortData",
    "load_battery_series",
    "load_test_circuits",
    "load_example_battery",
    "load_resonator",
    "sources_table",
]

_ZHANG = "6791a126659c1d11594cdf04bf17a483b26ecf68"
_IMPY = "6a269c4ef79b27cf52d1773fa6a8041af497cce1"
_SKRF = "5d1258a756359dbee4ce6d06aff7b65e9199b087"


def _gh(owner_repo: str, commit: str, path: str) -> List[str]:
    """Raw-file URLs of a pinned commit (primary + mirror)."""
    return [f"https://raw.githubusercontent.com/{owner_repo}/{commit}/{path}",
            f"https://github.com/{owner_repo}/raw/{commit}/{path}"]


@dataclass
class RemoteFile:
    local: str
    urls: List[str]
    sha256: str
    size: int


@dataclass
class Source:
    key: str
    title: str
    citation: str
    repository: str
    commit: str
    license: str
    files: Dict[str, RemoteFile] = field(default_factory=dict)


SOURCES: Dict[str, Source] = {
    "zhang2020": Source(
        key="zhang2020",
        title="Li-ion coin cell EIS over 598 cycles (cell 35C02, state V)",
        citation=("Zhang Y., Tang Q., Zhang Y., Wang J., Stimming U., Lee A. A. Identifying degradation "
                  "patterns of lithium ion batteries from impedance spectroscopy using machine learning. "
                  "Nature Communications 11, 1706 (2020). doi:10.1038/s41467-020-15235-7; "
                  "data doi:10.5281/zenodo.3633835"),
        repository="https://github.com/YunweiZhang/ML-identify-battery-degradation",
        commit=_ZHANG,
        license="data: CC BY 4.0 (Zenodo record); repository without explicit licence – not redistributed",
        files={"zhang2020_Code-Matlab.zip": RemoteFile(
            "zhang2020_Code-Matlab.zip",
            _gh("YunweiZhang/ML-identify-battery-degradation", _ZHANG, "Code-Matlab.zip"),
            "7d6ab9b3d48a2eadf3a4a737d8c6c8c35a81f0f1d2780249cbb0e735f5e0f69c", 3666132)},
    ),
    "impedancepy": Source(
        key="impedancepy",
        title="Physical R0-L-(R1||C1) test circuits (3 circuits x 2 replicates) and a Li-ion EIS spectrum",
        citation=("Murbach M. D., Gerwe B., Dawson-Elli N., Tsui L. impedance.py: A Python package for "
                  "electrochemical impedance analysis. Journal of Open Source Software 5(52), 2349 (2020). "
                  "doi:10.21105/joss.02349"),
        repository="https://github.com/ECSHackWeek/impedance.py",
        commit=_IMPY,
        license="MIT",
        files={},
    ),
    "skrf": Source(
        key="skrf",
        title="Stripline resonators 36/72/144 mm, S-parameters 1-5 GHz (Keysight N5242A)",
        citation=("Arsenovic A., Hillairet J., Anderson J., Forstén H., Rieß V., Eller M., Sauber N., "
                  "Weikle R., Barnhart W., Forstmayr F. scikit-rf: An Open Source Python Package for "
                  "Microwave Network Creation, Analysis, and Calibration. IEEE Microwave Magazine 23(1), "
                  "98-105 (2022). doi:10.1109/MMM.2021.3117139"),
        repository="https://github.com/scikit-rf/scikit-rf",
        commit=_SKRF,
        license="BSD-3-Clause",
        files={},
    ),
}

_IMPY_FILES = {
    "Circuit1_EIS_1.z": ("00c920ac9224ddeeb900a45374b6d44254587dac7baa24b3b56eaff8f52a857d", None),
    "Circuit1_EIS_2.z": ("aefc7bfa3c89253545661e0d55b6a7fe6d8ab480e70a230c4bc8bdaa6e8aae09", None),
    "Circuit2_EIS_1.z": ("2bb61593f88bc5cf22d274964d38dc69b4d6266e27c80cee4769f967b8c34b63", None),
    "Circuit2_EIS_2.z": ("351507e00a4d322e41dd0a83c057df9c5c7b643a65448ffcbe3d86e0473c8fb5", None),
    "Circuit3_EIS_1.z": ("3324a5ce6e9abb3550b2094d97f444ba1c9384e51b50e61448911f32b3c87d92", None),
    "Circuit3_EIS_2.z": ("93731fc0606c3038795c3d6f147c369579df4bd48446610bc908f44875190a6f", None),
    "exampleData.csv": ("7a6185b5e95e7457dc38a1e6bdc98e6c3a986fe9973592095020bab070fd47d2", None),
}
for _name, (_h, _) in _IMPY_FILES.items():
    SOURCES["impedancepy"].files[f"impedancepy_{_name}"] = RemoteFile(
        f"impedancepy_{_name}", _gh("ECSHackWeek/impedance.py", _IMPY, f"data/{_name}"), _h, -1)

_SKRF_FILES = {
    36: "9fd37c4871cc79aa5aac9934a0c77eac739225c1e2825a393fd7603804ffd1d7",
    72: "7d28c2928d7d0e7eed7fa84ae5e13687baf7119bb093e3b742f7bdfd0dd0943f",
    144: "fefeda1edca5643e4a29b87ff47bf754b8de8ad9cfff9c4ba08a777264ed667b",
}
for _mm, _h in _SKRF_FILES.items():
    SOURCES["skrf"].files[f"skrf_resonator_{_mm}mm.s2p"] = RemoteFile(
        f"skrf_resonator_{_mm}mm.s2p",
        _gh("scikit-rf/scikit-rf", _SKRF, f"doc/source/examples/qfactor/data/resonator_{_mm}mm.s2p"), _h, -1)


# --------------------------------------------------------------------------
# download & cache
# --------------------------------------------------------------------------
def data_dir(path: Optional[os.PathLike] = None) -> Path:
    """Cache directory: argument, ``$RLCFREQ_DATA_DIR`` or ``<repo>/data/raw``."""
    if path is not None:
        p = Path(path)
    elif os.environ.get("RLCFREQ_DATA_DIR"):
        p = Path(os.environ["RLCFREQ_DATA_DIR"])
    else:
        p = Path(__file__).resolve().parents[1] / "data" / "raw"
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256(path: os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, timeout: float = 60.0) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "rlcfreq-lab/1.0 (+https://github.com)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec - pinned HTTPS URLs
        data = r.read()
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".part-")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)


def fetch(local_name: str, cache: Optional[os.PathLike] = None, retries: int = 2,
          verbose: bool = False) -> Path:
    """Return the path of a verified local copy, downloading it if needed.

    Raises ``RuntimeError`` if no URL yields a file with the expected SHA-256.
    """
    rf = next((s.files[local_name] for s in SOURCES.values() if local_name in s.files), None)
    if rf is None:
        raise KeyError(f"unknown file {local_name!r}")
    dest = data_dir(cache) / local_name
    if dest.exists() and sha256(dest) == rf.sha256:
        return dest
    errors = []
    for url in rf.urls:
        for attempt in range(retries + 1):
            try:
                if verbose:
                    print(f"GET {url}")
                _download(url, dest)
                digest = sha256(dest)
                if digest == rf.sha256:
                    return dest
                errors.append(f"{url}: checksum {digest[:12]}… != {rf.sha256[:12]}…")
                dest.unlink(missing_ok=True)
                break
            except Exception as exc:  # network errors: retry, then next mirror
                errors.append(f"{url}: {exc}")
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"could not obtain {local_name}:\n  " + "\n  ".join(errors))


def fetch_all(cache: Optional[os.PathLike] = None, verbose: bool = True) -> Dict[str, Path]:
    out = {}
    for src in SOURCES.values():
        for name in src.files:
            out[name] = fetch(name, cache, verbose=verbose)
    return out


def sources_table():
    """Sources with URLs and checksums as a pandas table (for the report)."""
    import pandas as pd

    rows = []
    for s in SOURCES.values():
        for name, rf in s.files.items():
            rows.append(dict(source=s.key, file=name, url=rf.urls[0], sha256=rf.sha256[:16] + "…",
                             license=s.license.split(";")[0]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# parsers
# --------------------------------------------------------------------------
def parse_zplot(text: str) -> Dict[str, np.ndarray]:
    """Parse a Scribner ZPlot ``.z`` file (``Freq, Ampl, Bias, Time, Z', Z''``)."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if "End Comments" in l), None)
    if start is None:
        start = next(i for i, l in enumerate(lines) if "Freq(Hz)" in l)
    meta = {}
    for l in lines[:start]:
        m = re.match(r"\s*([^:]+):\s*(.*)$", l)
        if m:
            meta[m.group(1).strip()] = m.group(2).strip()
    rows = [re.split(r"\t|,\s*", l.strip()) for l in lines[start + 1:] if l.strip()]
    a = np.array([[float(v) for v in r[:6]] for r in rows])
    return dict(freq=a[:, 0], Z=a[:, 4] + 1j * a[:, 5], time=a[:, 3], amplitude=a[:, 1], meta=meta)


_UNITS = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}


def parse_touchstone(text: str, nports: int = 2) -> Dict[str, np.ndarray]:
    """Minimal Touchstone v1 reader (RI / MA / DB, any frequency unit).

    Returns ``freq`` (Hz), ``S`` with shape ``(N, nports, nports)`` and ``z0``.
    For two-port files the column order is S11 S21 S12 S22 (Touchstone v1).
    """
    unit, fmt, z0 = 1e9, "MA", 50.0
    values: List[float] = []
    comments = []
    for raw in text.splitlines():
        line = raw.split("!", 1)
        if raw.lstrip().startswith("!"):
            comments.append(raw.strip()[1:].strip())
        body = line[0].strip()
        if not body:
            continue
        if body.startswith("#"):
            tok = body[1:].split()
            low = [t.lower() for t in tok]
            for t in low:
                if t in _UNITS:
                    unit = _UNITS[t]
                if t in {"ri", "ma", "db"}:
                    fmt = t.upper()
            if "r" in low:
                z0 = float(tok[low.index("r") + 1])
            continue
        values.extend(float(v) for v in body.split())
    n = 1 + 2 * nports * nports
    a = np.array(values).reshape(-1, n)
    f = a[:, 0] * unit
    p = a[:, 1:].reshape(-1, nports * nports, 2)
    if fmt == "RI":
        S = p[..., 0] + 1j * p[..., 1]
    elif fmt == "MA":
        S = p[..., 0] * np.exp(1j * np.deg2rad(p[..., 1]))
    else:
        S = 10 ** (p[..., 0] / 20) * np.exp(1j * np.deg2rad(p[..., 1]))
    S = S.reshape(-1, nports, nports)
    if nports == 2:
        S = S.transpose(0, 2, 1)   # v1 2-port order is S11 S21 S12 S22 (column-major)
    return dict(freq=f, S=S, z0=z0, comments=comments)


def zhang_frequency_grid() -> np.ndarray:
    """Frequency grid of the Zhang et al. spectra (Hz), high to low.

    The processed files store 60 complex values without frequencies.  The
    paper states "60 frequencies between 0.02 Hz and 20 kHz"; frequencies
    quoted in the paper and follow-up work (115.778, 17.80, 11.145, 2.16 and
    0.165 Hz) fall on grid indices 22, 30, 32, 39, 50 of a logarithmic grid
    with both end points included, ``f_k = 2·10⁴ · 10^{−6k/59}``: they agree
    within their quoted precision plus 6.3·10⁻⁴ relative (tested).  This
    reconstruction is an explicit assumption of the analysis; a relative
    frequency error ε rescales the fitted L, C, Q by at most ~ε (α unaffected).
    """
    return 2e4 * 10 ** (-6 * np.arange(60) / 59)


# --------------------------------------------------------------------------
# containers & loaders
# --------------------------------------------------------------------------
@dataclass
class EISSpectrum:
    name: str
    freq: np.ndarray             # Hz
    Z: np.ndarray                # Ohm, complex (Z' + i Z'')
    time: Optional[np.ndarray] = None   # s, acquisition time of each point
    meta: Dict = field(default_factory=dict)

    @property
    def omega(self) -> np.ndarray:
        return 2 * np.pi * self.freq


@dataclass
class BatterySeries:
    name: str
    freq: np.ndarray             # Hz (60)
    cycles: np.ndarray           # cycle numbers
    Z: np.ndarray                # (n_cycles, 60) complex
    capacity: np.ndarray         # mAh

    @property
    def omega(self) -> np.ndarray:
        return 2 * np.pi * self.freq


@dataclass
class TwoPortData:
    name: str
    freq: np.ndarray
    S: np.ndarray
    z0: float
    length_mm: float

    @property
    def s21(self) -> np.ndarray:
        return self.S[:, 1, 0]


def load_battery_series(cache: Optional[os.PathLike] = None) -> BatterySeries:
    """Cell 35C02 of Zhang et al. (2020): 299 spectra, cycles 2, 4, …, 598.

    The stored second half of each row is ``−Im Z`` (positive for the
    capacitive arcs); it is converted to ``Z = Z' + i Z''``.
    """
    path = fetch("zhang2020_Code-Matlab.zip", cache)
    with zipfile.ZipFile(path) as z:
        E = np.loadtxt(io.BytesIO(z.read("Code-Matlab/EIS_data_35C02.txt")))
        cap = np.loadtxt(io.BytesIO(z.read("Code-Matlab/capacity35C02.txt")))
    if E.shape != (299, 120):
        raise ValueError(f"unexpected shape {E.shape}")
    Z = E[:, :60] - 1j * E[:, 60:]
    cycles = np.arange(2, 2 * len(E) + 1, 2)
    return BatterySeries("Li-ion coin cell 35C02 (Zhang et al. 2020)", zhang_frequency_grid(), cycles, Z, cap)


def load_test_circuits(cache: Optional[os.PathLike] = None) -> List[EISSpectrum]:
    """The six ZPlot spectra of the physical test circuits (impedance.py)."""
    out = []
    for c in (1, 2, 3):
        for r in (1, 2):
            path = fetch(f"impedancepy_Circuit{c}_EIS_{r}.z", cache)
            d = parse_zplot(path.read_text(encoding="utf8"))
            out.append(EISSpectrum(f"Circuit{c}_EIS_{r}", d["freq"], d["Z"], d["time"],
                                   dict(circuit=c, replicate=r, **d["meta"])))
    return out


def load_example_battery(cache: Optional[os.PathLike] = None) -> EISSpectrum:
    """Li-ion EIS spectrum shipped with impedance.py (``f, Z', Z''`` CSV)."""
    path = fetch("impedancepy_exampleData.csv", cache)
    a = np.loadtxt(path, delimiter=",")
    return EISSpectrum("impedance.py Li-ion example", a[:, 0], a[:, 1] + 1j * a[:, 2])


def load_resonator(length_mm: int, cache: Optional[os.PathLike] = None) -> TwoPortData:
    path = fetch(f"skrf_resonator_{length_mm}mm.s2p", cache)
    d = parse_touchstone(path.read_text(encoding="utf8"))
    return TwoPortData(f"stripline {length_mm} mm", d["freq"], d["S"], d["z0"], float(length_mm))
