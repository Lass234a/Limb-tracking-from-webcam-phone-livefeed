"""DAT-07 (+DOC-07, REP-03): each recording says which code, libraries, computer and protocol file produced it."""
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from limbtrack import provenance  # noqa: E402
from limbtrack.engine import META_SCHEMA, SOFTWARE_VERSION, Engine  # noqa: E402
from limbtrack.protocol import load_protocol  # noqa: E402
from limbtrack.synthetic import leg_points, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TESTS = load_protocol(ROOT / "protocol.json")


def test_the_meta_file_carries_the_provenance(tmp_path):
    eng = Engine(TESTS, recordings_dir=tmp_path)
    eng.protocol_path = ROOT / "protocol.json"
    eng.set_test("knee_extension")
    p = leg_points(hip_flex=90, knee_flex=70, ankle_df=5)
    eng.process(render(p, seed=0), 0.0)
    for n in eng.active_dots:
        assert eng.mark_dot(n, *(p[n] + 2))
    eng.set_source("demo_leg.mp4", {"is_file": True}, None)
    paths = eng.start_trial("P01")
    eng.process(render(p, seed=1), 0.1)
    eng.stop_trial()
    meta = json.loads(paths["meta.json"].read_text())
    assert meta["software_version"] == SOFTWARE_VERSION == "0.4.1" and meta["meta_schema"] == META_SCHEMA
    prov = meta["provenance"]
    assert len(prov["code_sha256"]) == 64 and len(prov["protocol_sha256_canonical"]) == 64
    assert prov["opencv"] and prov["numpy"] and prov["pyside6"] and prov["python"].startswith("3.")
    assert prov["git_commit"] is None or len(prov["git_commit"]) == 40


def test_no_git_gives_none_and_never_an_error(monkeypatch):
    def broken(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance.subprocess, "run", broken)
    provenance.collect.cache_clear()
    rec = provenance.collect(None)
    provenance.collect.cache_clear()
    assert rec["git_commit"] is None and rec["git_dirty"] is None and rec["code_sha256"]      # the content hash still identifies the code
    assert rec["protocol_sha256_canonical"] is None


def test_the_protocol_hash_ignores_spacing_and_line_endings(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text('{"x": 1,\n "y": [1, 2]}', encoding="utf-8")
    b.write_bytes(b'{"y":[1,2],\r\n"x":1}')
    assert provenance.protocol_sha256(a) == provenance.protocol_sha256(b)
    b.write_text('{"x": 2, "y": [1, 2]}', encoding="utf-8")
    assert provenance.protocol_sha256(a) != provenance.protocol_sha256(b)


def test_a_video_file_source_is_recorded_by_name_only():
    import limbtrack.camera as camera
    cam = object.__new__(camera.CameraSource)
    cam.source, cam.is_file = r"C:\Users\someone\Desktop\leg.mp4", True

    class Cap:
        def get(self, prop):
            return 1280

    cam.cap = Cap()
    assert cam.description == "leg.mp4 (1280x1280)" and "someone" not in cam.description
