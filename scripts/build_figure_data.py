"""Build `figure_data/` — the tidy tables every publication figure reads.

Each figure generator reads one or two files from here, so the dated run directories under
`results/` stay local build artifacts and `figure_data/` is the tracked form of the same numbers.

Build one experiment's tables, or all of them:

    uv run python scripts/build_figure_data.py --experiments exp1
    uv run python scripts/build_figure_data.py

Every table is validated before anything is published, and the whole set is staged in a temporary
directory and moved into place only once all of it passes. A partial or failed run therefore leaves
the existing tables untouched rather than replacing them with something incomplete.
"""

from __future__ import annotations
import argparse
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figure_data"
RESULTS = ROOT / "results"

MAMMAL = RESULTS / "2026-07-16_mammalian-orthologs-transcript_cdsmask"
STRAT = RESULTS / "2026-08-08_platypus-strat-400"
ARM = "stage4_cds_mean_blocks27"

# The residual-stream depth of Evo2-7B. Every per-block table must cover all of it; a table built
# from a subset would silently understate the layer profile the figures plot.
N_BLOCKS = 32
LAYERS = list(range(N_BLOCKS))

EXPERIMENTS = ("exp1", "exp2", "exp3")

# stage 4 names a condition from a non-reported layer with a trailing `_L<n>` (see
# gc_codon_figures.layer_suffix); the reported block-27 conditions carry no suffix.
_LAYER_SUFFIX = re.compile(r"_L\d+$")

WITHIN_BASELINES = {
    "kmer_within_family_correlations_angular.csv": (
        "k-mer composition (CDS, k=6)",
        "spearman_angular_kmer",
    ),
    "within_family_patristic_angular.csv": ("patristic tree", "spearman_angular_patristic"),
    "within_family_speciestree_angular.csv": (
        "species tree (mammal)",
        "spearman_angular_speciestree",
    ),
    "within_family_gc_angular.csv": ("GC content (control)", "spearman_angular_gc"),
}


class BuildError(SystemExit):
    """A source is missing or a built table failed validation; nothing is published."""


def _blocks(root: Path, rel: str) -> list[tuple[int, Path]]:
    """(layer, path) for every block, or fail naming the blocks that are missing."""
    found = [(L, root / f"blocks{L}" / rel) for L in LAYERS if (root / f"blocks{L}" / rel).exists()]
    if len(found) != N_BLOCKS:
        have = {L for L, _ in found}
        raise BuildError(
            f"{rel}: found {len(found)}/{N_BLOCKS} blocks under {root}\n"
            f"  missing layers: {sorted(set(LAYERS) - have)}\n"
            f"  the run that writes it has not finished; refusing to build a partial table"
        )
    return found


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise BuildError(f"missing source {path}\n  run the experiment that writes it first")
    return pd.read_csv(path)


# ── Experiment 1


def between_family_by_layer() -> pd.DataFrame:
    rows = []
    for layer, path in _blocks(MAMMAL, "between_family_ot_scores.csv"):
        d = pd.read_csv(path)
        d = d[d["approach"] == "wasserstein"]
        for r in d.itertuples():
            rows.append(
                {
                    "layer": layer,
                    "baseline": r.baseline,
                    "axis": r.axis,
                    "rho": r.spearman_rho,
                    "p_mantel": r.p_mantel,
                }
            )
    return pd.DataFrame(rows)


def within_family_by_layer() -> pd.DataFrame:
    rows = []
    for fname, (metric, col) in WITHIN_BASELINES.items():
        for layer, path in _blocks(MAMMAL, fname):
            for r in pd.read_csv(path).itertuples():
                rows.append(
                    {"layer": layer, "metric": metric, "family": r.family, "rho": getattr(r, col)}
                )
    return pd.DataFrame(rows)


# ── Experiment 2


def control_preservation() -> pd.DataFrame:
    """One long table over both axes. `family` is empty on between-family rows, whose geometry is
    a single family x family matrix per block."""
    rows = []
    for layer, path in _blocks(MAMMAL, "controls/control_between_scores.csv"):
        for r in pd.read_csv(path).itertuples():
            rows.append(
                {
                    "layer": layer,
                    "condition": r.condition,
                    "axis": "between_family",
                    "family": "",
                    "rho": r.rho_vs_natural_between,
                }
            )
    for layer, path in _blocks(MAMMAL, "controls/control_within_scores.csv"):
        for r in pd.read_csv(path).itertuples():
            rows.append(
                {
                    "layer": layer,
                    "condition": r.condition,
                    "axis": "within_family",
                    "family": r.family,
                    "rho": r.rho_angular_vs_natural,
                }
            )
    return pd.DataFrame(rows)


def control_identity() -> pd.DataFrame:
    """Per-rung nucleotide identity of each control to the gene it was built from.

    Sequence-level, written by `scripts/controls/control_sequence_identity.py --stage identity`.
    Kept beside the control geometry because figure 13 reads the two together: the question is
    whether the rho in figure 4 is just the identity in this table.
    """
    return _read(MAMMAL / "control_sequence_identity.csv")


def control_identity_by_family() -> pd.DataFrame:
    """The same identity per (rung, family) — the within-rung test compares families inside one
    rung, so it never relies on comparing rungs to each other."""
    return _read(MAMMAL / "control_sequence_identity_by_family.csv")


# ── Experiment 3


# Figures 6a/6b were rendered from the 103-gene family panel until 2026-09-01 and now come from
# the n=400 conservation-stratified panel, the same one every other exp3 figure uses. The retired
# panel's tables are archived under figure_data/old_paired103/ and its figures as "... (OLD)".
PANEL = "strat400"


def direction_layer_stats() -> pd.DataFrame:
    d = _read(STRAT / "geom_cds_mean" / "layer_stats.csv")
    d.insert(0, "panel", PANEL)
    return d


def direction_per_gene_by_layer() -> pd.DataFrame:
    d = _read(STRAT / "geom_cds_mean" / "per_gene_by_layer.csv")
    d.insert(0, "panel", PANEL)
    return d


# The manuscript intervention is at block 27. The stage-4 run directory also retains exploratory
# block-24 conditions (`add_a1.0_L24` and friends), which is why the layer is filtered on the
# recorded provenance rather than on the condition name: a name-based rule would silently admit
# any future condition that did not happen to carry the suffix.
STEER_LAYER = "blocks.27"

# The block-27 conditions the manuscript reports. Named rather than counted so that a run which
# silently loses an arm, or gains one, fails here instead of reaching a figure.
#
# The manuscript describes one per-gene random direction reused across the dose ladder, so the
# norm-matched null is required at EVERY reported dose. stage4_steer already reuses the direction
# across whatever alphas it is handed; the gap was in the driver, whose C3 stage ran `--arms add`
# alone. Until that stage is re-run with `--arms add random`, this build fails naming the two
# missing cells rather than publishing a table whose ladder is unmatched above alpha 2.
STEER_CONDITIONS = (
    "unsteered",
    "add_own",
    "add_a0.5",
    "add_a1.0",
    "add_a2.0",
    "add_a3.0",
    "add_a4.0",
    "random_a0.5",
    "random_a1.0",
    "random_a2.0",
    "random_a3.0",
    "random_a4.0",
    "cross_gene_a1.0",
    "add_cone_removed_a1.0",
    "add_gc_removed_a1.0",
)


# Conditions the run generated that are deliberately NOT published. Excluded here rather than
# deleted from the stage-4 tables, so the run directory stays a faithful record of what was
# executed while the tracked artifact carries only what the manuscript reports. Anything that is
# neither expected nor listed here still fails the panel check, so this cannot quietly absorb an
# unrecognised arm.
STEER_EXCLUDED = {
    "panel_same_k5": "cluster-panel arm (H2c); stage3_gates recorded tier C as NOT licensed",
    "panel_other_k5": "cluster-panel arm (H2c); stage3_gates recorded tier C as NOT licensed",
}


def steering_outcomes() -> pd.DataFrame:
    """One row per (gene, condition), for the block-27 intervention only.

    The saved table has one row per generated sample; figures 7 and 8 both average over samples
    within a gene before plotting, so this is lossless for them. Columns constant inside every cell
    are carried through unchanged, the rest are averaged.
    """
    d = _read(STRAT / ARM / "stage4_scores_nt.csv")
    # `unsteered` is the hook-free baseline: no layer is touched, so it records layer "none" with
    # n_layers 0 rather than a block. It belongs to every layer's comparison, including this one.
    hook_free = d["n_layers"].eq(0)
    keep = d["layer"].eq(STEER_LAYER) | hook_free
    dropped = sorted(set(d.loc[~keep, "condition"]))
    if not keep.any():
        raise BuildError(
            f"no rows at layer {STEER_LAYER} in {STRAT / ARM / 'stage4_scores_nt.csv'}\n"
            f"  layers present: {sorted(set(d['layer']))}"
        )
    if (bad := d.loc[hook_free & d["layer"].ne("none"), "condition"].unique()).size:
        raise BuildError(f"hook-free rows carry a layer: {sorted(bad)}")
    d = d[keep].reset_index(drop=True)
    if excluded := sorted(set(d["condition"]) & set(STEER_EXCLUDED)):
        print(f"  steering_outcomes: excluding {excluded} ({STEER_EXCLUDED[excluded[0]]})")
        d = d[~d["condition"].isin(STEER_EXCLUDED)].reset_index(drop=True)
    if set(d["condition"]) != set(STEER_CONDITIONS):
        have = set(d["condition"])
        raise BuildError(
            f"block-27 conditions do not match the reported panel\n"
            f"  missing:   {sorted(set(STEER_CONDITIONS) - have)}\n"
            f"  unexpected: {sorted(have - set(STEER_CONDITIONS))}"
        )
    if dropped:
        print(
            f"  steering_outcomes: dropped {len(dropped)} non-{STEER_LAYER} conditions: {dropped}"
        )
    keys = ["gene", "condition"]
    g = d.groupby(keys, sort=False)
    numeric = [c for c in d.columns if c not in keys and pd.api.types.is_numeric_dtype(d[c])]
    other = [c for c in d.columns if c not in keys and c not in numeric]
    constant = [c for c in numeric if (g[c].nunique(dropna=False) <= 1).all()]
    averaged = sorted(set(numeric) - set(constant) - {"sample"})
    agg = {c: "first" for c in constant + other}
    agg.update({c: "mean" for c in averaged})
    out = g.agg(agg).reset_index().drop(columns=["sample"], errors="ignore")
    return out[[c for c in d.columns if c in out.columns]]


# ── the table registry


@dataclass(frozen=True)
class Spec:
    """One output table: which experiment owns it, how to build it, and what must hold."""

    experiment: str
    description: str
    build: Callable[[], pd.DataFrame]
    columns: tuple[str, ...] = ()
    all_layers: bool = False  # a `layer` column that must cover every block
    unique: tuple[str, ...] = ()  # these columns must form a unique key
    min_rows: int = 1
    counts: dict[str, int] = field(
        default_factory=dict
    )  # column -> exact number of distinct values
    # Reject any condition naming a non-reported intervention layer. stage 4 suffixes a
    # condition with `_L<n>` for every layer except the reported one, so this catches a table
    # built from a cache that spans layers -- which is how block-24 rows reached Figure 10's
    # tracked inputs. Tables carrying an explicit `layer` column are filtered instead, at build.
    one_layer: bool = False
    # (key, panel): every value of `key` must carry EVERY value of `panel`. A global distinct
    # count cannot catch a gene that is missing one arm, because the other genes still supply
    # the level -- and a figure averaging over an incomplete cell reports a different comparison
    # from the one its caption claims.
    complete: tuple[str, str] | None = None


TABLES: dict[str, Spec] = {
    "exp1_between_family_by_layer": Spec(
        "exp1",
        "Figure 1 — W2 family geometry vs baselines",
        between_family_by_layer,
        columns=("layer", "baseline", "rho"),
        all_layers=True,
        unique=("layer", "baseline"),
        counts={"baseline": 3},
    ),
    "exp1_within_family_by_layer": Spec(
        "exp1",
        "Figures 2, 3 — within-group rho vs baselines",
        within_family_by_layer,
        columns=("layer", "metric", "family", "rho"),
        all_layers=True,
        unique=("layer", "metric", "family"),
        counts={"metric": len(WITHIN_BASELINES)},
    ),
    "exp2_control_preservation": Spec(
        "exp2",
        "Figures 4, 12 — control vs natural geometry",
        control_preservation,
        columns=("layer", "condition", "axis", "family", "rho"),
        all_layers=True,
        unique=("layer", "condition", "axis", "family"),
        counts={"axis": 2},
    ),
    "exp2_control_identity": Spec(
        "exp2",
        "Figure 13 — control identity to source vs the rho it produces",
        control_identity,
        columns=("condition", "pos_identity", "null_pos_identity", "pos_excess", "n_seqs"),
        unique=("condition",),
        counts={"condition": 6},
    ),
    "exp2_control_identity_by_family": Spec(
        "exp2",
        "Figure 13 — the same identity per family",
        control_identity_by_family,
        columns=("condition", "family", "pos_identity", "n_seqs"),
        unique=("condition", "family"),
        counts={"condition": 6, "family": 48},
    ),
    "exp3_direction_layer_stats": Spec(
        "exp3",
        "Figures 6a, 6b — per-layer direction statistics",
        direction_layer_stats,
        columns=("panel", "layer", "loo_median", "delta_norm_cv"),
        all_layers=True,
        unique=("panel", "layer"),
        counts={"panel": 1},
    ),
    "exp3_direction_per_gene_by_layer": Spec(
        "exp3",
        "Figures 6b, 11 — per-gene direction geometry",
        direction_per_gene_by_layer,
        columns=("panel", "layer", "gene", "loo_cos", "delta_norm"),
        all_layers=True,
        unique=("panel", "layer", "gene"),
        counts={"panel": 1},
    ),
    "exp3_panel": Spec(
        "exp3",
        "Figure 5 — the 400 human/platypus pairs",
        lambda: _read(STRAT / "stage1" / "pairs.csv"),
        columns=("gene", "stratum", "perc_id_hp"),
        unique=("gene",),
        counts={"stratum": 5},
    ),
    "exp3_panel_attrition": Spec(
        "exp3",
        "Figure 5 — candidates dropped by QC",
        lambda: _read(STRAT / "stage1" / "attrition.csv"),
        columns=("gene_id", "reason"),
    ),
    "exp3_panel_coverage": Spec(
        "exp3",
        "Figure 5 — aligned CDS coverage per pair",
        lambda: _read(STRAT / "stage2" / "aligned_coverage.csv"),
        columns=("gene", "retained_frac"),
        unique=("gene",),
    ),
    "exp3_steering_outcomes": Spec(
        "exp3",
        "Figures 7, 8 — steering outcomes per gene and condition",
        steering_outcomes,
        columns=("gene", "condition", "stratum", "pct_private_bp_correct", "aa_id_to_target"),
        unique=("gene", "condition"),
        counts={"condition": len(STEER_CONDITIONS)},
        complete=("gene", "condition"),
    ),
    "exp3_site_directionality_summary": Spec(
        "exp3",
        "Figures 9a, 9b — leave-human and platypus-choice rates",
        lambda: _read(STRAT / "site_directionality" / "summary.csv"),
        columns=("site_set", "condition", "d_L", "d_C"),
        unique=("site_set", "condition"),
    ),
    "exp3_site_directionality_gc_class": Spec(
        "exp3",
        "Figures 9a, 9b — the same, split by GC class",
        lambda: _read(STRAT / "site_directionality" / "gc_class.csv"),
        columns=("site_set", "condition", "gc_class"),
        unique=("site_set", "condition", "gc_class"),
        counts={"gc_class": 3},
    ),
    "exp3_site_directionality_per_gene": Spec(
        "exp3",
        "Figures 9a, 9b — per-gene spread",
        lambda: _read(STRAT / "site_directionality" / "per_gene.csv"),
        columns=("gene", "condition", "site_set", "excess"),
        unique=("gene", "condition", "site_set"),
    ),
    "exp3_generation_composition": Spec(
        "exp3",
        "Figure 10 — GC of the generations",
        lambda: _read(STRAT / "figures" / "12b_composition_gene_means.csv"),
        columns=("gene", "condition", "gc", "gc3"),
        unique=("gene", "condition"),
        one_layer=True,
    ),
    "exp3_generation_reference_windows": Spec(
        "exp3",
        "Figure 10 — human and platypus reference GC",
        lambda: _read(STRAT / "figures" / "12c_reference_windows.csv"),
        columns=("gene", "gc_human", "gc_platypus"),
        unique=("gene",),
    ),
    "exp3_codon_substitutions": Spec(
        "exp3",
        "Figure 10 — GC by codon position",
        lambda: _read(STRAT / "figures" / "13b_codon_substitution_stats.csv"),
        columns=("gene", "condition", "gc1", "gc2", "gc3"),
        one_layer=True,
    ),
    "exp3_rates_tree_stats": Spec(
        "exp3",
        "Figure 11 — fixed-topology branch lengths",
        lambda: _read(STRAT / "stage5" / "tree_stats.csv"),
        columns=("gene", "status"),
        unique=("gene",),
    ),
    "exp3_rates_dnds": Spec(
        "exp3",
        "Figure 11 — dN, dS and omega",
        lambda: _read(STRAT / "stage5" / "dnds.csv"),
        columns=("gene", "status", "dN_hp_yn", "dS_hp_yn"),
        unique=("gene",),
    ),
    "exp3_rates_vs_direction": Spec(
        "exp3",
        "Figure 11 — rate vs direction geometry",
        lambda: _read(STRAT / "stage5" / "rate_vs_direction.csv"),
        columns=("predictor", "outcome", "rho"),
    ),
    "exp3_rates_vs_direction_shape": Spec(
        "exp3",
        "Figure 11 — the same, shape tests",
        lambda: _read(STRAT / "stage5" / "rate_vs_direction_shape.csv"),
        columns=("predictor", "outcome"),
    ),
    "exp3_rates_vs_gain": Spec(
        "exp3",
        "Figure 11 — rate vs steering gain",
        lambda: _read(STRAT / "stage5" / "rate_vs_gain.csv"),
        columns=("predictor", "condition", "spearman_rho"),
    ),
}

COPIES: dict[str, tuple[str, str, Path]] = {
    "exp3_direction_nulls.npz": (
        "exp3",
        "Figures 6a, 6b — permutation nulls",
        STRAT / "geom_cds_mean" / "null_distributions.npz",
    ),
}


def validate(name: str, spec: Spec, table: pd.DataFrame) -> None:
    """Fail loudly rather than publish a table the figures would silently under-plot."""
    missing = [c for c in spec.columns if c not in table.columns]
    if missing:
        raise BuildError(f"{name}: missing columns {missing}")
    if len(table) < spec.min_rows:
        raise BuildError(f"{name}: {len(table)} rows, expected at least {spec.min_rows}")
    if spec.all_layers:
        have = set(table["layer"].unique())
        if have != set(LAYERS):
            raise BuildError(
                f"{name}: layer column covers {len(have)}/{N_BLOCKS} blocks; "
                f"missing {sorted(set(LAYERS) - have)}"
            )
    if spec.unique:
        dup = table.duplicated(subset=list(spec.unique)).sum()
        if dup:
            raise BuildError(f"{name}: {dup} duplicate rows on key {spec.unique}")
    for column, expected in spec.counts.items():
        n = table[column].nunique()
        if n != expected:
            raise BuildError(f"{name}: {column} has {n} distinct values, expected {expected}")
    if spec.one_layer and "condition" in table.columns:
        suffixed = sorted(
            {c for c in table["condition"].dropna().unique() if _LAYER_SUFFIX.search(str(c))}
        )
        if suffixed:
            raise BuildError(
                f"{name}: {len(suffixed)} conditions name another intervention layer: {suffixed}\n"
                "  the source cache spans layers; rebuild it with "
                "gc_codon_figures.py --emit-tables"
            )
    if spec.complete:
        key, panel = spec.complete
        levels = set(table[panel].dropna().unique())
        short = {
            k: sorted(levels - set(vals))
            for k, vals in table.groupby(key)[panel]
            if set(vals) != levels
        }
        if short:
            shown = list(short.items())[:5]
            raise BuildError(
                f"{name}: {len(short)}/{table[key].nunique()} {key}s do not carry the full "
                f"{panel} panel ({len(levels)} levels).\n"
                + "\n".join(f"    {k}: missing {v}" for k, v in shown)
                + ("\n    ..." if len(short) > len(shown) else "")
            )
    for column in spec.unique:
        if table[column].isna().any():
            raise BuildError(f"{name}: null values in key column {column!r}")


README_START = "<!-- TABLE: generated by build_figure_data.py; edit the descriptions in TABLES -->"
README_END = "<!-- /TABLE -->"


def write_readme_table(out_dir: Path, manifest: pd.DataFrame) -> None:
    """Rewrite the README's table block from the manifest.

    The block used to be maintained by hand and had drifted: it advertised row counts from an
    earlier build. A reader checking a tracked artifact against its own documentation should not
    be the thing that discovers that, so the counts are now written from the same rows the
    manifest gets. Prose outside the markers is left alone.
    """
    readme = out_dir / "README.md"
    if not readme.exists():
        return
    text = readme.read_text()
    if README_START not in text or README_END not in text:
        return  # a README without the markers is fully hand-written; leave it be
    lines = ["| File | Rows x cols | Size | Contents |", "|---|---|---|---|"]
    for r in manifest.itertuples():
        shape = "binary" if r.rows == 0 else f"{r.rows:,} x {r.columns}"
        size = f"{r.bytes / 1024:.0f} KB" if r.bytes < 1048576 else f"{r.bytes / 1048576:.1f} MB"
        lines.append(f"| `{r.file}` | {shape} | {size} | {r.description} |")
    head, _, rest = text.partition(README_START)
    _, _, tail = rest.partition(README_END)
    readme.write_text(f"{head}{README_START}\n" + "\n".join(lines) + f"\n{README_END}{tail}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENTS,
        default=list(EXPERIMENTS),
        help="which experiments' tables to build (default: all). Each runner builds only its own, "
        "so an experiment can be run on a machine that has no outputs from the other two.",
    )
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    wanted = set(args.experiments)
    specs = {n: s for n, s in TABLES.items() if s.experiment in wanted}
    copies = {n: c for n, c in COPIES.items() if c[0] in wanted}
    print(f"building {len(specs) + len(copies)} tables for {', '.join(sorted(wanted))}")

    # Stage everything first: a table that fails validation must not replace a good one.
    staged: list[tuple[str, int, int, str]] = []
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        tmpdir = Path(tmp)
        for name, spec in specs.items():
            table = spec.build()
            validate(name, spec, table)
            table.to_csv(tmpdir / f"{name}.csv", index=False)
            staged.append((f"{name}.csv", len(table), len(table.columns), spec.description))
            print(f"  {len(table):>7,} x {len(table.columns):<3} {name}.csv")
        for name, (_exp, desc, src) in copies.items():
            if not src.exists():
                raise BuildError(f"missing source {src}")
            shutil.copy2(src, tmpdir / name)
            staged.append((name, 0, 0, desc))
            print(f"  {'(binary)':>11} {name}")

        args.out.mkdir(parents=True, exist_ok=True)
        for filename, *_ in staged:
            shutil.move(str(tmpdir / filename), args.out / filename)

    # Merge into the manifest so building one experiment does not drop the others' rows.
    manifest = args.out / "MANIFEST.csv"
    rows = {
        r.file: dict(
            file=r.file, rows=r.rows, columns=r.columns, bytes=r.bytes, description=r.description
        )
        for r in (pd.read_csv(manifest).itertuples() if manifest.exists() else [])
    }
    for filename, n_rows, n_cols, desc in staged:
        rows[filename] = dict(
            file=filename,
            rows=n_rows,
            columns=n_cols,
            bytes=(args.out / filename).stat().st_size,
            description=desc,
        )
    order = [f"{n}.csv" for n in TABLES] + list(COPIES)
    out = pd.DataFrame([rows[f] for f in order if f in rows])
    out.to_csv(manifest, index=False)
    write_readme_table(args.out, out)
    print(
        f"\n{len(staged)} published, {len(out)} in manifest, "
        f"{out.bytes.sum() / 1048576:.1f} MiB -> {args.out}"
    )


if __name__ == "__main__":
    main()
