"""Data layer: parsers, HTTP download with checksums, frequency grid.

Network tests are opt-in: ``RLCFREQ_NETWORK=1 pytest -m network``.
"""
import hashlib
import http.server
import os
import socketserver
import threading
from pathlib import Path

import numpy as np
import pytest

from rlcfreq import data

ZPLOT = """ZPLOT2 ASCII
  Measured Data, ZPlot:       3.2c
  Data Points:                3
  Freq(Hz)\tAmpl\tBias\tTime(Sec)\tZ'(a)\tZ''(b)\tGD\tErr\tRange
End Comments
5.000000E+04\t1.0000E-02\t0.0000E+00\t2.590000E+00\t2.9036E+01\t6.3662E-01\t0.0000E+00\t0\t4
1.000000E+02\t1.0000E-02\t0.0000E+00\t9.000000E+00\t5.0000E+01\t-1.0000E+01\t0.0000E+00\t0\t4
1.000000E+00\t1.0000E-02\t0.0000E+00\t4.085000E+01\t7.5803E+01\t-1.6244E-01\t0.0000E+00\t0\t4
"""

TOUCH_RI = """! test
# GHz S RI R 50.0
!freq ReS11 ImS11 ReS21 ImS21 ReS12 ImS12 ReS22 ImS22
1.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
2.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0
"""

TOUCH_MA = """# hz S ma R 50
1e9 1.0 0.0 0.5 90.0 0.5 90.0 1.0 180.0
"""


def test_parse_zplot():
    d = data.parse_zplot(ZPLOT)
    assert np.allclose(d["freq"], [5e4, 1e2, 1.0])
    assert np.isclose(d["Z"][0], 29.036 + 0.63662j)
    assert np.allclose(d["time"], [2.59, 9.0, 40.85])


def test_parse_touchstone_ri_order_and_units():
    d = data.parse_touchstone(TOUCH_RI)
    assert np.allclose(d["freq"], [1e9, 2e9])
    S = d["S"][0]
    assert np.isclose(S[0, 0], 0.1 + 0.2j) and np.isclose(S[1, 0], 0.3 + 0.4j)   # S21
    assert np.isclose(S[0, 1], 0.5 + 0.6j) and np.isclose(S[1, 1], 0.7 + 0.8j)   # S12, S22
    assert d["z0"] == 50.0


def test_parse_touchstone_ma():
    d = data.parse_touchstone(TOUCH_MA)
    assert np.isclose(d["S"][0, 1, 0], 0.5j) and np.isclose(d["S"][0, 1, 1], -1.0)


def test_zhang_frequency_grid_matches_published_values():
    """Frequencies quoted in the literature lie on the reconstructed grid."""
    f = data.zhang_frequency_grid()
    assert len(f) == 60 and np.isclose(f[0], 2e4) and np.isclose(f[-1], 0.02)
    published = {22: "115.778", 30: "17.80", 32: "11.145", 39: "2.16", 50: "0.165"}
    for k, text in published.items():
        v = float(text)
        half_ulp = 0.5 * 10 ** (-len(text.split(".")[1]))     # rounding of the quoted value
        assert abs(f[k] - v) <= half_ulp + 1e-3 * v


def test_registry_is_pinned_and_complete():
    names = [n for s in data.SOURCES.values() for n in s.files]
    assert len(names) == 11
    for s in data.SOURCES.values():
        for rf in s.files.values():
            assert len(rf.sha256) == 64
            assert s.commit in rf.urls[0] and rf.urls[0].startswith("https://")


@pytest.fixture()
def http_server(tmp_path):
    """Local HTTP server (stand-in for GitHub) serving one file."""
    root = tmp_path / "srv"
    root.mkdir()
    (root / "file.z").write_text(ZPLOT)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def log_message(self, *a):
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
        th = threading.Thread(target=httpd.serve_forever, daemon=True)
        th.start()
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
        httpd.shutdown()


def test_fetch_downloads_verifies_and_caches(http_server, tmp_path, monkeypatch):
    digest = hashlib.sha256(ZPLOT.encode()).hexdigest()
    rf = data.RemoteFile("test_file.z", [http_server + "/missing.z", http_server + "/file.z"], digest, -1)
    monkeypatch.setitem(data.SOURCES["impedancepy"].files, "test_file.z", rf)
    cache = tmp_path / "cache"
    p = data.fetch("test_file.z", cache, retries=0)        # first URL 404 -> mirror
    assert p.read_text() == ZPLOT and data.sha256(p) == digest
    p.write_text(ZPLOT)                                    # cached: no network needed
    rf.urls = ["http://127.0.0.1:9/never"]
    assert data.fetch("test_file.z", cache, retries=0) == p


def test_fetch_rejects_wrong_checksum(http_server, tmp_path, monkeypatch):
    rf = data.RemoteFile("bad.z", [http_server + "/file.z"], "0" * 64, -1)
    monkeypatch.setitem(data.SOURCES["impedancepy"].files, "bad.z", rf)
    with pytest.raises(RuntimeError, match="checksum"):
        data.fetch("bad.z", tmp_path, retries=0)
    assert not (tmp_path / "bad.z").exists()


def _cached(name):
    p = data.data_dir() / name
    return p.exists()


@pytest.mark.skipif(not _cached("zhang2020_Code-Matlab.zip"), reason="data not downloaded (make data)")
def test_battery_series_shape_and_sign_convention():
    b = data.load_battery_series()
    assert b.Z.shape == (299, 60) and b.cycles[0] == 2 and b.cycles[-1] == 598
    assert np.all(b.Z[:, 0].imag > 0)            # inductive at 20 kHz
    assert np.all(b.Z[:, 25].imag < 0)           # capacitive arc in the middle
    assert b.capacity[0] > b.capacity[-1]        # capacity fade


@pytest.mark.skipif(not _cached("impedancepy_Circuit1_EIS_1.z"), reason="data not downloaded (make data)")
def test_test_circuits_replicates_agree():
    sp = {s.name: s for s in data.load_test_circuits()}
    for c in (1, 2, 3):
        a, b = sp[f"Circuit{c}_EIS_1"], sp[f"Circuit{c}_EIS_2"]
        assert np.allclose(a.freq, b.freq)
        assert np.max(np.abs(a.Z - b.Z) / np.abs(a.Z)) < 0.02


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("RLCFREQ_NETWORK") != "1", reason="set RLCFREQ_NETWORK=1")
def test_real_downloads(tmp_path):
    paths = data.fetch_all(tmp_path, verbose=False)
    assert len(paths) == 11
