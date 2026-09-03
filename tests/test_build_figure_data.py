"""The figure_data builder must refuse to publish an incomplete or malformed table."""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_figure_data as bfd  # noqa: E402


def _layer_table(layers=None, baselines=("pfam_jsd", "kmer", "gc_content")) -> pd.DataFrame:
    layers = bfd.LAYERS if layers is None else layers
    return pd.DataFrame(
        [
            {"layer": layer, "baseline": b, "axis": "x", "rho": 0.5, "p_mantel": None}
            for layer in layers
            for b in baselines
        ]
    )


SPEC = bfd.TABLES["exp1_between_family_by_layer"]


def test_missing_layers_are_rejected():
    """The failure mode that matters: a run that stopped early must not become a shorter table."""
    partial = _layer_table(layers=range(20))
    with pytest.raises(SystemExit) as e:
        bfd.validate("t", SPEC, partial)
    assert "20/32" in str(e.value)
    assert "31" in str(e.value)


def test_missing_columns_are_rejected():
    with pytest.raises(SystemExit, match="missing columns"):
        bfd.validate("t", SPEC, _layer_table().drop(columns=["rho"]))


def test_duplicate_keys_are_rejected():
    doubled = pd.concat([_layer_table(), _layer_table()], ignore_index=True)
    with pytest.raises(SystemExit, match="duplicate rows"):
        bfd.validate("t", SPEC, doubled)


def test_blocks_reports_the_layers_it_could_not_find(tmp_path):
    for layer in range(30):
        d = tmp_path / f"blocks{layer}"
        d.mkdir()
        (d / "scores.csv").write_text("a\n1\n")
    with pytest.raises(SystemExit) as e:
        bfd._blocks(tmp_path, "scores.csv")
    assert "30/32" in str(e.value)
    assert "[30, 31]" in str(e.value)


# ── the build is all-or-nothing
#
# The failure this guards against: the builder used to write each table straight into figure_data/
# as it went, and _blocks() filtered on existence rather than requiring all 32. A machine holding
# only one experiment's run directories therefore OVERWROTE the other experiments' tracked tables
# with empty or short ones, then crashed on the first hard-missing input -- and the damage only
# surfaced later as an empty figure. Recovery needed `git restore figure_data/`.


def _spec(build, experiment="exp1"):
    return bfd.Spec(experiment, "test table", build, columns=("layer",))


def _good():
    return pd.DataFrame({"layer": [0, 1], "value": [1.0, 2.0]})


def test_a_failed_build_publishes_nothing(tmp_path, monkeypatch):
    """A table that validated before the failure must not reach the output directory."""
    out = tmp_path / "figure_data"
    out.mkdir()
    sentinel = out / "good.csv"
    sentinel.write_text("tracked,content\n1,2\n")

    def explode():
        raise bfd.BuildError("simulated missing input")

    monkeypatch.setattr(bfd, "TABLES", {"good": _spec(_good), "bad": _spec(explode)})
    monkeypatch.setattr(bfd, "COPIES", {})
    monkeypatch.setattr(sys, "argv", ["build_figure_data.py", "--out", str(out)])
    with pytest.raises(SystemExit):
        bfd.main()
    assert sentinel.read_text() == "tracked,content\n1,2\n", "a good table was overwritten"
    assert sorted(p.name for p in out.iterdir()) == ["good.csv"], "a partial build was published"


def test_a_successful_build_publishes_everything(tmp_path, monkeypatch):
    """The counterpart: when nothing fails, every staged table lands."""
    out = tmp_path / "figure_data"
    out.mkdir()
    monkeypatch.setattr(bfd, "TABLES", {"a": _spec(_good), "b": _spec(_good)})
    monkeypatch.setattr(bfd, "COPIES", {})
    monkeypatch.setattr(sys, "argv", ["build_figure_data.py", "--out", str(out)])
    bfd.main()
    assert sorted(p.name for p in out.iterdir()) == ["MANIFEST.csv", "a.csv", "b.csv"]
