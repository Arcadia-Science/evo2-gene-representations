"""Contracts the publication path must hold, independent of any run directory.

Each test here pins a decision that is invisible in the numbers: an ordering, a layer filter, a
cache's right to be reused. All three failed silently before -- a permuted family order still
produces a plausible rho, a layer-24 row still averages into a layer-27 cell, and a stale reduced
triangle is just a vector of floats. Fast and embedding-free.
"""

from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))

import build_figure_data as bfd  # noqa: E402
from gene_families import HUMAN_FAMILY_ORDER  # noqa: E402
from ot_between_family import MIN_MEMBERS, W2_CONTRACT_VERSION, family_order  # noqa: E402

# ── 1. the W2 family-order contract comes from the manifest, not a centroid artifact


def _labels(families, n=MIN_MEMBERS):
    return [f for f in families for _ in range(n)]


def test_family_order_follows_the_curated_panel_order():
    """Not alphabetical: the order is the curated panel order, which is what the run records."""
    picked = ["hox", "globins", "opsins"]
    got = family_order(_labels(picked))
    assert got == [f for f in HUMAN_FAMILY_ORDER if f in picked]
    assert got != sorted(picked), "curated order must not collapse to alphabetical"


def test_family_order_is_independent_of_label_arrangement():
    """The manifest's row order must not leak into the family order."""
    picked = ["globins", "hox", "opsins"]
    assert family_order(_labels(picked)) == family_order(list(reversed(_labels(picked))))


def test_family_order_drops_families_below_the_member_floor():
    labels = _labels(["globins", "hox"]) + ["opsins"] * (MIN_MEMBERS - 1)
    assert set(family_order(labels)) == {"globins", "hox"}


def test_family_order_rejects_a_manifest_that_has_drifted():
    """A family the curated panel does not define fails loudly, not silently ordered last."""
    with pytest.raises(ValueError, match="absent from the curated panel"):
        family_order(_labels(["globins", "not_a_real_family"]))


def test_family_order_needs_no_centroid_file():
    """The point of the fix: nothing on the W2 path READS a centroid/geodesic distance matrix.

    Checked against the artifact name rather than the word, so the comment explaining why the
    dependency was removed does not itself trip the test.
    """
    src = (ROOT / "scripts" / "baselines" / "ot_between_family_sweep.py").read_text()
    assert "evo2_mammal_centroid_distances" not in src
    # `from geodesic_utils import mantel_test, upper_triangle` is fine and deliberately allowed:
    # both are metric-agnostic helpers that happen to live in a badly named module.
    code = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in code if "centroid" in ln.lower()], "centroid referenced in code"


def test_reordering_families_consistently_leaves_rho_unchanged():
    """Why the reorder is provenance and not a change to Figure 1: a Spearman rho over the upper
    triangle is invariant to a permutation applied to BOTH matrices."""
    rng = np.random.default_rng(0)
    f = 12
    a = rng.random((f, f))
    b = rng.random((f, f))
    a, b = a + a.T, b + b.T
    np.fill_diagonal(a, 0)
    np.fill_diagonal(b, 0)
    iu = np.triu_indices(f, 1)
    before = spearmanr(a[iu], b[iu]).statistic
    perm = rng.permutation(f)
    after = spearmanr(a[np.ix_(perm, perm)][iu], b[np.ix_(perm, perm)][iu]).statistic
    assert before == pytest.approx(after, abs=1e-12)


# ── 2. steering outcomes carry one intervention layer


def _scores(conditions, genes=("A", "B"), samples=2, layer=bfd.STEER_LAYER):
    rows = []
    for gene in genes:
        for cond in conditions:
            for s in range(samples):
                free = cond == "unsteered"
                rows.append(
                    {
                        "gene": gene,
                        "condition": cond,
                        "sample": s,
                        "layer": "none" if free else layer,
                        "n_layers": 0 if free else 1,
                        "stratum": 1,
                        "pct_private_bp_correct": 40.0 + s,
                        "aa_id_to_target": 0.5,
                    }
                )
    return pd.DataFrame(rows)


def _patch_scores(monkeypatch, tmp_path, df):
    path = tmp_path / "stage4_scores_nt.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(bfd, "_read", lambda _p: pd.read_csv(path))


def test_layer24_rows_are_excluded(monkeypatch, tmp_path):
    """The reported intervention is block 27; retained block-24 rows must not enter."""
    both = pd.concat(
        [_scores(bfd.STEER_CONDITIONS), _scores(["add_a1.0_L24"], layer="blocks.24")],
        ignore_index=True,
    )
    _patch_scores(monkeypatch, tmp_path, both)
    out = bfd.steering_outcomes()
    assert set(out.condition) == set(bfd.STEER_CONDITIONS)
    assert (out.layer != "blocks.24").all()


def test_layer_filter_is_on_provenance_not_the_condition_name(monkeypatch, tmp_path):
    """A block-24 row whose name lacks the _L24 suffix must still be dropped."""
    both = pd.concat(
        [_scores(bfd.STEER_CONDITIONS), _scores(["add_a1.0"], genes=("C",), layer="blocks.24")],
        ignore_index=True,
    )
    _patch_scores(monkeypatch, tmp_path, both)
    assert "C" not in set(bfd.steering_outcomes().gene)


def test_aggregation_over_samples_is_lossless(monkeypatch, tmp_path):
    """One row per (gene, condition), valued as the mean over that cell's samples (40..43)."""
    _patch_scores(monkeypatch, tmp_path, _scores(bfd.STEER_CONDITIONS, samples=4))
    out = bfd.steering_outcomes()
    assert len(out) == 2 * len(bfd.STEER_CONDITIONS)
    assert not out.duplicated(subset=["gene", "condition"]).any()
    assert out.pct_private_bp_correct.tolist() == pytest.approx([41.5] * len(out))


def test_a_missing_arm_fails_the_condition_panel(monkeypatch, tmp_path):
    _patch_scores(monkeypatch, tmp_path, _scores(bfd.STEER_CONDITIONS[:-1]))
    with pytest.raises(bfd.BuildError, match="do not match the reported panel"):
        bfd.steering_outcomes()


def test_a_gene_missing_one_dose_fails_completeness():
    """A global distinct count cannot catch this: the other gene still supplies every level."""
    full = _scores(bfd.STEER_CONDITIONS)
    short = full[~((full.gene == "B") & (full.condition == "add_a4.0"))]
    table = short.drop(columns=["sample"]).drop_duplicates(subset=["gene", "condition"])
    with pytest.raises(bfd.BuildError, match="do not carry the full"):
        bfd.validate("exp3_steering_outcomes", bfd.TABLES["exp3_steering_outcomes"], table)


def test_the_random_null_covers_every_reported_dose():
    """The manuscript describes one per-gene random direction reused across the dose ladder, so
    the norm-matched null must exist at EVERY dose -- not merely at some of them. This is the
    contract that was violated while the ladder ran to alpha 4 and the null stopped at alpha 2."""
    doses = {c.removeprefix("add_") for c in bfd.STEER_CONDITIONS if c.startswith("add_a")}
    randoms = {c.removeprefix("random_") for c in bfd.STEER_CONDITIONS if c.startswith("random_a")}
    assert doses, "no dose ladder declared"
    assert randoms == doses, f"unmatched doses: {sorted(doses ^ randoms)}"


def test_the_driver_runs_the_random_arm_at_every_dose():
    """The gap was never in stage4_steer, which reuses the direction across whatever alphas it is
    handed -- it was in the driver stage that asked for `add` alone."""
    src = (ROOT / "experiments" / "exp3_platypus_steering.sh").read_text()
    for line in src.splitlines():
        if "--arms" in line and "--alphas" in line and "add" in line:
            assert "random" in line, f"dose stage omits the matched null: {line.strip()}"


# ── 3. a reduced-triangle cache may only be reused when it says what it is

CACHE_FIELDS = (
    "cache_schema",
    "w2_contract",
    "panel_id",
    "metric",
    "between_metric",
    "fam_order",
    "group_order",
    "offsets",
    "between_len",
)


def test_the_scorer_validates_every_provenance_field():
    """A reduced triangle stays numerically plausible under the wrong family order, so every
    field that fixes the coordinate system is checked before reuse, not just the panel hash."""
    src = (ROOT / "scripts" / "mammalian_orthologs" / "mammal_controls_score.py").read_text()
    for field in CACHE_FIELDS:
        assert f'"{field}"' in src, f"cache contract does not mention {field}"
    assert 'CACHE_SCHEMA = "controls-cache-' in src
    assert "W2_CONTRACT_VERSION" in src


def test_w2_contract_version_is_recorded_in_result_metadata():
    """A written W2 matrix must say which contract produced it."""
    from ot_between_family import compute_ot_matrices

    rng = np.random.default_rng(0)
    emb = rng.normal(size=(12, 8))
    fams = ["a", "b", "c"]
    labels = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4)
    res = compute_ot_matrices(emb, labels, fams, alphas=(), solver="exact")
    assert res.meta["w2_contract"] == W2_CONTRACT_VERSION
    assert res.meta["fam_order"] == fams


def test_the_controls_scorer_never_reads_a_positional_w2_artifact():
    """Two scripts build W2 family triangles: the sweep (curated order) and the controls scorer
    (its own alphabetical `fams`). A triangle is indexed by POSITION, so comparing one script's
    output against the other's would silently mis-pair families. The controls scorer avoids this
    by computing natural and control triangles in-process with one order; nothing may reintroduce
    a cross-script read of a positional artifact.
    """
    src = (ROOT / "scripts" / "mammalian_orthologs" / "mammal_controls_score.py").read_text()
    for artifact in ("betweenfam_ot", "between_family_ot_scores", "evo2_mammal_centroid_distances"):
        assert artifact not in src, f"controls scorer reads positional artifact {artifact}"


# ── 4. exp3's stage order matches its data dependencies

EXP3_DRIVER = ROOT / "experiments" / "exp3_platypus_steering.sh"


def _exp3_stage_order() -> list[tuple[str, str]]:
    """(label, command) for each `stage` line in the exp3 driver, in file order."""
    lines = EXP3_DRIVER.read_text().splitlines()
    out = []
    for i, ln in enumerate(lines):
        if ln.startswith("stage "):
            label = ln.split('"')[3]
            out.append((label, lines[i + 1] if i + 1 < len(lines) else ""))
    return out


def test_the_ortholog_stage_runs_after_stage1_builds_the_panel():
    """private_site_orthologs reads the panel and the platypus CDS that stage1_qc writes, so
    ordering it first made a full --run fail on its very first data stage."""
    order = [lab for lab, _ in _exp3_stage_order()]
    cmds = {lab: cmd for lab, cmd in _exp3_stage_order()}
    qc = next(i for i, lab in enumerate(order) if "stage1_qc.py" in cmds[lab])
    orth = next(i for i, lab in enumerate(order) if "private_site_orthologs.py" in cmds[lab])
    assert qc < orth, f"stage1_qc must precede private_site_orthologs (got {qc} vs {orth})"


def test_the_ortholog_stage_reads_only_stage1_outputs():
    """Pins why the ordering above is required: every input this stage opens comes from stage 1."""
    src = (ROOT / "scripts" / "steering" / "platypus" / "strat" / "private_site_orthologs.py").read_text()
    reads = [ln for ln in src.splitlines() if "args.run /" in ln]
    assert reads, "expected the stage to read from the run directory"
    assert all('"stage1"' in ln for ln in reads), reads


def test_no_script_depends_on_a_gene_symbols_file_nothing_writes():
    """gene_symbols.tsv was read by one line and written by none; the symbols it carried were
    discarded by fetch_homologies, which keys on the Ensembl gene id whenever one is supplied.

    Code lines only, so the comment recording why the dependency was removed does not trip this.
    """
    hits = []
    for path in (ROOT / "scripts").rglob("*.py"):
        for i, ln in enumerate(path.read_text().splitlines(), 1):
            if "gene_symbols" in ln and not ln.lstrip().startswith("#"):
                hits.append(f"{path.relative_to(ROOT)}:{i}")
    assert not hits, f"gene_symbols is referenced in code at {hits}"


# Scripts that resolve their score column through strat_metric. Their default metric is `private`,
# whose column exists only after the nucleotide rescore -- and no --metric setting avoids that,
# because the platy-bp column comes from the same rescore. Only the deprecated `legacy` metric
# predates it, so any of these ordered before the rescore exits and stops the whole chain.
RESCORE = "stage4_rescore_nt.py"


def _metric_consumers() -> set[str]:
    strat = ROOT / "scripts" / "steering" / "platypus" / "strat"
    return {p.name for p in strat.glob("*.py") if "strat_metric" in p.read_text()}


def test_the_nucleotide_rescore_precedes_every_metric_consumer():
    text = EXP3_DRIVER.read_text()
    lines = text.splitlines()
    rescore_at = next(i for i, ln in enumerate(lines) if RESCORE in ln and not ln.lstrip().startswith("#"))
    consumers = _metric_consumers() - {RESCORE, "strat_metric.py"}
    seen = 0
    for name in sorted(consumers):
        for i, ln in enumerate(lines):
            if name in ln and not ln.lstrip().startswith("#"):
                seen += 1
                assert i > rescore_at, (
                    f"{name} runs at driver line {i + 1}, before the rescore at "
                    f"line {rescore_at + 1}; it would exit on the missing metric column"
                )
    assert seen, "no metric consumer found in the exp3 driver -- has it been restructured?"


def test_the_private_metric_column_comes_only_from_the_rescore():
    """Why the ordering matters, pinned against the writers rather than the driver."""
    strat = ROOT / "scripts" / "steering" / "platypus" / "strat"
    rescore = (strat / RESCORE).read_text()
    steer = (strat / "stage4_steer.py").read_text()
    for col in ("pct_private_bp_correct", "pct_diagnostic_correct"):
        assert f'"{col}"' in rescore, f"{col} should be written by the rescore"
        assert col not in steer, f"{col} unexpectedly written by stage4_steer"


# ── 5. Figure 10's tables are materialised by a stage, and carry one intervention layer

GC_FIGURES = ROOT / "scripts" / "steering" / "platypus" / "strat" / "gc_codon_figures.py"
FIG10_TABLES = (
    "12b_composition_gene_means.csv",
    "12c_reference_windows.csv",
    "13b_codon_substitution_stats.csv",
)


def test_a_driver_stage_materialises_figure_10s_tables_before_the_build():
    """These three tables used to appear only as a side effect of rendering the cross-layer
    diagnostics, which the driver never invoked -- so a clean --run reached build_figure_data
    with them missing."""
    lines = EXP3_DRIVER.read_text().splitlines()
    emit = next(
        (i for i, ln in enumerate(lines) if "--emit-tables" in ln and not ln.lstrip().startswith("#")),
        None,
    )
    assert emit is not None, "no stage materialises the composition/codon tables"
    build = next(i for i, ln in enumerate(lines) if "build_figure_data.py" in ln and "--experiments exp3" in ln)
    assert emit < build, "the tables must be emitted before figure_data is built"


def test_the_cached_tables_cover_only_the_reported_layer():
    """KEEP is what the emitter writes. Widening it back to the `_L24` ladder is what leaked
    block-24 rows into the tracked Figure 10 inputs."""
    src = GC_FIGURES.read_text()
    keep_line = next(ln for ln in src.splitlines() if ln.startswith("KEEP = "))
    assert "_L24" not in keep_line, keep_line
    assert "dose_ladder(" not in keep_line, f"KEEP must not span layers: {keep_line}"


def test_the_figure_10_specs_reject_another_layers_conditions():
    for name in ("exp3_generation_composition", "exp3_codon_substitutions"):
        assert bfd.TABLES[name].one_layer, f"{name} should be guarded by one_layer"
    spec = bfd.TABLES["exp3_codon_substitutions"]
    bad = pd.DataFrame(
        {
            "gene": ["A", "A"],
            "condition": ["add_a1.0", "add_a1.0_L24"],
            "gc1": [0.5, 0.5],
            "gc2": [0.5, 0.5],
            "gc3": [0.5, 0.5],
        }
    )
    with pytest.raises(bfd.BuildError, match="name another intervention layer"):
        bfd.validate("exp3_codon_substitutions", spec, bad)


def test_the_retired_diagnostics_are_archived_not_deleted():
    archived = (
        ROOT
        / "deprecated"
        / "gc-codon-diagnostics-2026-09-03"
        / "scripts/steering/platypus/strat/gc_codon_diagnostic_figures.py"
    )
    assert archived.exists(), "the retired figures must remain recoverable"
    body = archived.read_text()
    for fn in ("fig12", "fig13", "fig15", "fig16", "fig17", "panel_pn_ps"):
        assert f"def {fn}(" in body, f"{fn} missing from the archive"
    manifest = (ROOT / "deprecated" / "MANIFEST.tsv").read_text()
    assert "gc-codon-diagnostics-2026-09-03" in manifest, "archive not recorded in MANIFEST.tsv"


# ── 6. no dataset location is pinned to one machine

import paths  # noqa: E402

# `/opt/dlami/nvme` is the AWS instance store on the authors' box: absolute, absent everywhere
# else, usually unwritable under /opt, and EPHEMERAL -- it is wiped when the instance stops, which
# is how the platypus sequence set and 23 of the 24 genomes were lost. Nothing may depend on it.
MACHINE_PATHS = ("/opt/dlami", "/home/ubuntu", "/mnt/", "/scratch/")


def test_no_script_hardcodes_a_machine_specific_path():
    offenders = []
    for path in (ROOT / "scripts").rglob("*.py"):
        if path.name == "paths.py":  # documents the old locations in its docstring
            continue
        for i, ln in enumerate(path.read_text().splitlines(), 1):
            code = ln.split("#")[0]
            if any(m in code for m in MACHINE_PATHS):
                offenders.append(f"{path.relative_to(ROOT)}:{i}: {ln.strip()}")
    assert not offenders, "machine-specific paths:\n  " + "\n  ".join(offenders)


def test_every_dataset_root_defaults_inside_the_checkout():
    """A fresh clone must resolve somewhere writable, with no environment set."""
    for name in ("SCRATCH", "MAMMAL_GENOMES", "STRAT_SEQS", "STRAT_MAMMAL_CDS"):
        p = getattr(paths, name)
        assert ROOT in p.parents or p == ROOT, f"{name} defaults outside the repo: {p}"


def test_dataset_roots_are_git_ignored():
    """They hold multi-gigabyte downloads; the default must not make them committable."""
    import subprocess

    r = subprocess.run(
        ["git", "check-ignore", str(paths.SCRATCH)], cwd=ROOT, capture_output=True, text=True
    )
    assert r.returncode == 0, f"{paths.SCRATCH} is not git-ignored"


def test_environment_variables_relocate_each_root(monkeypatch):
    import importlib

    monkeypatch.setenv("GLM_SCRATCH", "/tmp/glm_scratch_test")
    reloaded = importlib.reload(paths)
    try:
        assert str(reloaded.MAMMAL_GENOMES) == "/tmp/glm_scratch_test/mammal_genomes"
        assert str(reloaded.STRAT_MAMMAL_CDS) == "/tmp/glm_scratch_test/strat_seqs/mammals"
        monkeypatch.setenv("GLM_MAMMAL_GENOMES", "/tmp/elsewhere")
        reloaded = importlib.reload(paths)
        assert str(reloaded.MAMMAL_GENOMES) == "/tmp/elsewhere", "per-dataset override must win"
    finally:
        monkeypatch.undo()
        importlib.reload(paths)


def test_a_missing_dataset_names_itself_and_the_variable(capsys):
    with pytest.raises(SystemExit) as e:
        paths.require(ROOT / "no" / "such" / "file", "the widget set", "download it", "GLM_WIDGETS")
    msg = str(e.value)
    assert "the widget set" in msg and "download it" in msg and "GLM_WIDGETS" in msg


def test_the_documented_variables_are_the_ones_the_code_reads():
    doc = (ROOT / "REPRODUCING.md").read_text()
    src = (ROOT / "scripts" / "paths.py").read_text()
    used = set(re.findall(r'os\.environ\.get\("(GLM_[A-Z_]+)"', src))
    assert used, "paths.py reads no GLM_* variable"
    missing = sorted(v for v in used if v not in doc)
    assert not missing, f"undocumented in REPRODUCING.md: {missing}"


# ── 7. external tools are declared, checked early, and documented

import check_tools  # noqa: E402


def test_every_external_tool_is_documented():
    """The table in REPRODUCING.md and the registry the code uses must not drift apart."""
    doc = (ROOT / "REPRODUCING.md").read_text()
    missing = [t.name for t in check_tools.TOOLS if f"`{t.name}`" not in doc]
    assert not missing, f"undocumented external tools: {missing}"


def test_each_tool_declares_where_it_is_used_and_how_to_install_it():
    for t in check_tools.TOOLS:
        assert t.experiments, f"{t.name} names no experiment"
        assert set(t.experiments) <= {"exp1", "exp2", "exp3"}, t.experiments
        assert t.stage and t.install and t.used_version, f"{t.name} is under-specified"


def test_the_missing_tool_message_is_actionable():
    """One wording everywhere: what is missing, what needs it, and how to get it."""
    for t in check_tools.TOOLS:
        msg = check_tools.missing_message(t.name)
        assert t.name in msg
        assert t.install in msg, f"{t.name}: message omits the install command"
        assert t.used_version in msg, f"{t.name}: message omits the version used"


def test_a_missing_tool_exits_rather_than_returning_none():
    original = check_tools.BY_NAME["mafft"]
    try:
        check_tools.BY_NAME["mafft"] = check_tools.Tool(
            "definitely-not-a-real-tool", ("exp1",), "nothing", "nowhere", "0", on_path=True
        )
        with pytest.raises(SystemExit):
            check_tools.require("mafft")
    finally:
        check_tools.BY_NAME["mafft"] = original


def test_every_runner_preflights_its_own_tools():
    """Checking hours in is the failure this guards against; --figures must not be gated."""
    for exp in ("exp1", "exp2", "exp3"):
        driver = next((ROOT / "experiments").glob(f"{exp}_*.sh")).read_text()
        assert f"preflight_tools {exp}" in driver, f"{exp} does not preflight"
    helpers = (ROOT / "experiments" / "_helpers.sh").read_text()
    body = helpers.split("preflight_tools()", 1)[1]
    assert '[ "$MODE" = run ]' in body, "preflight must be skipped outside --run"


def test_the_stages_that_shell_out_check_their_tools_first():
    """Each of these invoked a binary through subprocess with no prior check, so a machine without
    it raised FileNotFoundError partway through instead of saying what to install."""
    expected = {
        "steering/platypus/strat/stage5_dnds.py": ("codeml", "yn00", "trimal"),
        "steering/platypus/strat/stage5_trees.py": ("mafft", "iqtree2", "trimal"),
        "baselines/protein_alignment_patristic.py": ("mafft", "FastTree"),
    }
    for rel, tools in expected.items():
        src = (ROOT / "scripts" / rel).read_text()
        assert "check_tools" in src, f"{rel} does not use the tool registry"
        for tool in tools:
            assert f'"{tool}"' in src, f"{rel} never requires {tool}"
