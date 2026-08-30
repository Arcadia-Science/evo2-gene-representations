"""Which site-recovery metric the platypus-strat figures plot — one definition, five scripts."""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

# Legacy protein-alignment scores remain available only for reproducibility.
LEGACY_COL = "pct_private_correct"

SCORES_NT = "stage4_scores_nt.csv"  # written by stage4_rescore_nt.py
SCORES_LEGACY = "stage4_scores.csv"  # written by stage4_steer.py

DEFAULT_METRIC = "private"

# "private" requires uniqueness among sampled mammals; "platy_not_human" is pairwise.
# Keep the legacy column name unchanged to prevent it from being mistaken for the strict metric.
METRICS: dict[str, dict[str, str]] = {
    "private": {
        "col": "pct_autapomorphy_correct",
        "n_col": "n_autapomorphy_scorable",
        "delta_label": "private-bp gain",
        "level_label": "private bp correct",
        "axis_delta": "Δ private bp recovered (pp)",
        "axis_level": "private bp recovered (%)",
        "short": "private bp\n(pp)",
        "strip": "private base pairs\nrecovered",
        "gloss": "platypus base differs from human and no other sampled mammal carries it",
    },
    # Retained only for reproducing protein-alignment outputs.
    "legacy": {
        "col": LEGACY_COL,
        "n_col": "n_diag_scorable",
        "delta_label": "legacy platy-bp (not human) gain",
        "level_label": "legacy platy bp (not human) correct",
        "axis_delta": "Δ legacy platy bp (not human) recovered (pp)",
        "axis_level": "legacy platy bp (not human) recovered (%)",
        "short": "legacy platy bp\n(pp)",
        "strip": "legacy platy base pairs (not human)\nrecovered",
        "gloss": (
            "PRE-FIX pairwise human-vs-platypus difference, scored through the protein "
            "alignment whose bias runs along the conservation axis -- historical only. Its "
            "column is named pct_private_correct but it is NOT the `private` metric above"
        ),
    },
    "platy_not_human": {
        "col": "pct_diagnostic_correct",
        "n_col": "n_diagnostic_scorable",
        "delta_label": "platy-bp (not human) gain",
        "level_label": "platy bp (not human) correct",
        "axis_delta": "Δ platy bp (not human) recovered (pp)",
        "axis_level": "platy bp (not human) recovered (%)",
        "short": "platy bp (not human)\n(pp)",
        "strip": "platy base pairs (not human)\nrecovered",
        "gloss": "platypus base differs from human (pairwise; no uniqueness requirement)",
    },
}

# Pre-rename selector keys, still accepted so existing commands and shell drivers keep working.
ALIASES = {"autapomorphy": "private", "diagnostic": "platy_not_human"}

RESCORE_CMD = (
    "uv run python scripts/steering/platypus/strat/stage4_rescore_nt.py \\\n"
    "      --run {run} --dir {arm_dir}"
)


def add_metric_args(ap) -> None:
    """Attach the metric-selection flags. Every strat figure script takes the same three."""
    ap.add_argument(
        "--metric",
        choices=sorted(METRICS) + sorted(ALIASES),
        default=DEFAULT_METRIC,
        help="site set to plot (default private: the run's headline readout). "
        "`autapomorphy` and `diagnostic` are accepted as pre-rename aliases of "
        "`private` and `platy_not_human`",
    )
    ap.add_argument(
        "--scores",
        default=None,
        help=f"score-table filename inside the arm dir (default: {SCORES_NT} when it "
        f"exists, else {SCORES_LEGACY})",
    )
    ap.add_argument(
        "--min-voters",
        type=int,
        default=1,
        help="blank the private-bp metric for genes voted on by fewer than this many "
        "ortholog species (default 1: any positive evidence). Genes with no "
        "orthologs are unscorable regardless and are always reported.",
    )


def resolve_scores_path(run: Path, arm_dir: str, scores: str | None = None) -> Path:
    """Prefer the rescored table, so a figure picks up the nt-alignment fix without a flag."""
    d = Path(run) / arm_dir
    if scores:
        return d / scores
    nt = d / SCORES_NT
    return nt if nt.exists() else d / SCORES_LEGACY


def load_scores(
    run: Path,
    arm_dir: str,
    *,
    scores: str | None = None,
    metric: str = DEFAULT_METRIC,
    min_voters: int = 1,
    quiet: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Load a stage-4 score table and guarantee the requested metric is present and named `metric`.
    """
    spec = dict(METRICS[ALIASES.get(metric, metric)])
    path = resolve_scores_path(run, arm_dir, scores)
    if not path.exists():
        sys.exit(f"no score table at {path}\n  run the steering arm first, or pass --scores")
    df = pd.read_csv(path)

    if spec["col"] not in df.columns:
        legacy = " (this table predates the split and only has the legacy "
        legacy += f"`{LEGACY_COL}`)" if LEGACY_COL in df.columns else ""
        sys.exit(
            f"{path.name} has no `{spec['col']}` column{legacy}.\n"
            f"  The {metric} site set comes from the nucleotide rescore. Produce it with:\n\n    "
            + RESCORE_CMD.format(run=run, arm_dir=arm_dir)
            + "\n\n"
            f"  (that needs data/platypus_strat_orthologs/cds/ from autapomorphy_orthologs.py).\n"
            f"  To plot the other site set meanwhile, pass --metric "
            f"{'diagnostic' if metric == 'autapomorphy' else 'autapomorphy'}."
        )

    df = df.copy()
    df["metric"] = df[spec["col"]]

    # ---- evidence base -------------------------------------------------------------------------
    # A gene is UNSCORABLE for this metric when it has no scorable site of that kind: no orthologs
    # voted (autapomorphy), or the alignment yielded no site at all. stage4_rescore_nt leaves those
    # NaN rather than counting every diagnostic site as private, so the NaN is correct — but it is
    # invisible to a `groupby().mean()`, which is how an uneven evidence base slips into a figure.
    voters = None
    if "n_voting_species" in df.columns:
        voters = df.drop_duplicates("gene").set_index("gene")["n_voting_species"]
        if min_voters > 1:
            weak = set(voters[voters < min_voters].index)
            df.loc[df.gene.isin(weak), "metric"] = float("nan")

    per_gene = df.groupby("gene")["metric"].apply(lambda x: x.notna().any())
    scorable = sorted(per_gene[per_gene].index)
    unscorable = sorted(per_gene[~per_gene].index)
    report = {
        "path": path,
        "metric": metric,
        "n_genes": int(per_gene.size),
        "n_scorable": len(scorable),
        "unscorable": unscorable,
        "min_voters": min_voters,
        "voters_median": float(voters.median()) if voters is not None else None,
        "voters_min": int(voters.min()) if voters is not None else None,
        "voters_max": int(voters.max()) if voters is not None else None,
    }
    spec["report"] = report

    if not quiet:
        if metric == "legacy":
            print(
                "[metric] *** LEGACY COLUMN -- historical reproduction only. This is the "
                "pre-2026-08-18 pairwise number whose own control read 9.3% in the most-diverged "
                "stratum vs 0.7% in the most-conserved, where it must be 0. Do not report it as "
                "a new result. ***"
            )
        print(f"[metric] {metric} ({spec['col']}) from {path.name}")
        print(f"[metric] {spec['gloss']}")
        if voters is not None:
            print(
                f"[metric] voting species per gene: median {report['voters_median']:.0f}, "
                f"min {report['voters_min']}, max {report['voters_max']}"
                + (f"  (genes under {min_voters} voters blanked)" if min_voters > 1 else "")
            )
        if unscorable:
            shown = ", ".join(unscorable[:8]) + (" ..." if len(unscorable) > 8 else "")
            print(
                f"[metric] WARNING {len(unscorable)}/{report['n_genes']} genes have NO "
                f"{metric} evidence and drop out of every panel: {shown}"
            )
    return df, spec


def coverage_note(spec: dict) -> str:
    """One-line provenance for a figure caption — which site set, over how many genes, how voted."""
    r = spec["report"]
    note = f"{r['metric']}: {spec['gloss']}.  {r['n_scorable']}/{r['n_genes']} genes scorable"
    if r["voters_median"] is not None:
        note += f"; median {r['voters_median']:.0f} voting species per gene"
    if r["unscorable"]:
        note += f"; {len(r['unscorable'])} with no ortholog evidence excluded"
    return note + "."


def stratum_counts(df: pd.DataFrame, genes=None) -> pd.Series:
    """Per-stratum count of genes that actually carry a non-NaN metric value."""
    d = df if genes is None else df[df.gene.isin(set(genes))]
    ok = d[d["metric"].notna()].drop_duplicates("gene")
    return ok.groupby("stratum").size()
