"""Embed the matched-human panel's composition controls through Evo2 and score
control-vs-natural preservation."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "scripts" / "evo2"))
sys.path.insert(0, str(_REPO / "scripts" / "baselines"))
import arcadia_style as acs  # noqa: E402
from control_tables import write_control_table  # noqa: E402

# Reuse the exact embedding op + tap from the natural pipeline (shared Evo2 engine).
from evo2_embedding import EMBED_LAYER  # noqa: E402
from geodesic_utils import (  # noqa: E402
    between_preservation_rho,
    compute_centroid_geodesic,
    compute_geodesic,
    find_min_connected_k,
    within_preservation_rho,
)

ALL_CONTROLS = [
    "dinuc_shuffle",
    "codon_shuffle",
    "synonymous_recode",
    "gc_match",
    "kmer4_shuffle",
    "kmer6_shuffle",
    "missense_subset",
]
# Controls applicable to a genomic span (no reading frame): the matched-human panel set.
HUMAN_CONTROLS = ["gc_match", "dinuc_shuffle", "kmer4_shuffle", "kmer6_shuffle"]
# On CDS input the reading frame exists, so the protein/codon controls the cross-kingdom pipeline
# uses become meaningful too: codon_shuffle (scrambles protein, keeps codon counts),
# synonymous_recode (keeps protein, resamples synonymous codons) and its nonsynonymous counterpart
# missense_subset (edits a strict SUBSET of the recode's changed bases, damaging the protein).
HUMAN_CDS_CONTROLS = HUMAN_CONTROLS + ["codon_shuffle", "synonymous_recode", "missense_subset"]
# Display order/labels/colours for the human-panel summary figure. Composition gradient:
# 6-mer (most preserved) → 4-mer → codon → dinucleotide → GC-only, then the nested pair:
# synonymous_recode (protein kept) and missense_subset, which edits a strict SUBSET of the recode's
# changed bases — fewer nucleotides moved, protein damaged. Colours match control_ladder.
# figure_human_controls filters this to the controls actually present, so genomic runs (4 controls)
# render unchanged.
_C = acs.CONTROL_COLORS
HUMAN_CONTROL_DISPLAY = [
    ("kmer6_shuffle", "6-mer shuffle", _C["kmer6_shuffle"]),
    ("kmer4_shuffle", "4-mer shuffle", _C["kmer4_shuffle"]),
    ("codon_shuffle", "Codon shuffle", _C["codon_shuffle"]),
    ("dinuc_shuffle", "Dinucleotide shuffle", _C["dinuc_shuffle"]),
    ("gc_match", "GC-matched random", _C["gc_match"]),
    ("synonymous_recode", "Synonymous recode (protein kept)", _C["synonymous_recode"]),
    ("missense_subset", "Missense at recode sites (protein damaged)", _C["missense_subset"]),
]


def resolve_latest_run(pattern: str) -> Path | None:
    """Return the newest run whose directory name matches the layer suffix exactly."""
    import glob as _glob

    hits = sorted(_glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(hits[0]) if hits else None


def figure_human_controls(out_dir: Path) -> None:
    """Summary figure for the Evo2 matched-HUMAN panel controls, reading the consolidated
    control_within_scores.csv (+ control_between_scores.csv) in `out_dir`.
    """
    import matplotlib

    matplotlib.use("Agg")
    from plot_utils import control_preservation_figure  # scripts/ already on sys.path

    within = pd.read_csv(out_dir / "control_within_scores.csv")
    between_p = out_dir / "control_between_scores.csv"
    between = pd.read_csv(between_p) if between_p.exists() else None
    # Family order: largest families first (stable across controls).
    fam_order = (
        within.drop_duplicates("family")
        .sort_values("n_members", ascending=False)["family"]
        .tolist()
    )
    within_by = {
        k: dict(zip(g["family"], g["rho_geodesic_vs_natural"], strict=False))
        for k, g in within.groupby("condition")
    }
    between_by = (
        dict(zip(between["condition"], between["rho_vs_natural_centroid"], strict=False))
        if between is not None
        else None
    )
    # Render only the controls actually scored (genomic runs = 4; CDS runs = 6).
    present = set(within["condition"])
    display = [(k, lbl, c) for k, lbl, c in HUMAN_CONTROL_DISPLAY if k in present]
    ok = control_preservation_figure(
        out_dir / "control_preservation",
        fam_order,
        display,
        within_by,
        between_by,
        title_left="Evo2-human composition controls: is the within-family geometry preserved?",
    )
    if ok:
        print(f"  Saved {out_dir}/control_preservation.{{pdf,png}}")
    else:
        print("  [figure] no recognised controls in control_within_scores.csv — skipping figure")


def _human_setup(
    controls: list[str], device: str, force_reembed: bool, input_mode: str = "genomic"
):
    """Load the human panel and cached control embeddings."""
    # Import lazily — these pull in Ensembl/panel machinery not needed for --figure-only.
    import embed_and_geodesic_paralog as ev  # noqa: E402
    import sample_human_genes as ss  # noqa: E402

    # CDS panel is Evo2-only and needs no genomic locus, so load it the SAME way the CDS embedder
    # did (require_both=False, resolve_loci=False) — matches the natural run's 1122 genes exactly
    # and
    # avoids per-gene Ensembl REST. Genomic keeps the matched-both, locus-resolved behavior.
    genes, fams, locus_of, fam_order = ss.load_matched_panel(
        None, require_both=(input_mode != "cds"), resolve_loci=(input_mode != "cds")
    )
    if input_mode == "cds":
        # Same gene subset the CDS embedder keeps (genes with a cached canonical CDS), same order.
        cds = json.loads(Path("data/cache/cds_sequences.json").read_text())
        keep = [(g, f) for g, f in zip(genes, fams, strict=False) if g in cds]
        genes, fams = [g for g, _ in keep], [f for _, f in keep]
        fam_order = [f for f in fam_order if f in set(fams)]
        seqs_natural = {g: cds[g] for g in genes}
        base = ev.CACHE_DIR.parent / "evo2_human_layer_sweep_cds"
    else:
        seqs_natural = ev.load_or_fetch_genomic(genes, locus_of, ev.DEFAULT_MAX_LEN)
        base = ev.CACHE_DIR  # data/cache/evo2_human_layer_sweep
    fams_arr = np.array(fams)
    fam_of = dict(zip(genes, fams, strict=False))
    stacks: dict[str, np.ndarray] = {}
    for control in controls:
        print(f"\n=== loading {control} layer stack ({input_mode}) ===")
        seqs = ev.apply_control(seqs_natural, control, fam_of=fam_of)
        cache_dir = base.parent / f"{base.name}_{control}"
        stacks[control] = ev.run_sweep(
            genes,
            fams,
            seqs,
            device,
            force_reembed,
            50,
            ev.EVO2_WINDOW,
            cache_dir=cache_dir,
            control=control,
        )
    return stacks, genes, fams_arr, fam_order


def _human_patristic_by_family(
    genes: list[str], fam_of: dict[str, str], cache: Path = Path("data/cache/human_patristic")
):
    """Load each family's members and patristic submatrix from cache."""
    import json as _json

    out: dict[str, tuple[list[str], np.ndarray]] = {}
    for fam in sorted(set(fam_of.values())):
        npy, idsj = cache / f"{fam}.npy", cache / f"{fam}.ids.json"
        if not (npy.exists() and idsj.exists()):
            continue
        pos = {g: i for i, g in enumerate(_json.loads(idsj.read_text()))}
        fam_genes = [g for g in genes if fam_of.get(g) == fam and g in pos]
        if len(fam_genes) < 4:
            continue
        idx = [pos[g] for g in fam_genes]
        out[fam] = (fam_genes, np.load(npy)[np.ix_(idx, idx)])
    return out


def _patristic_recovery(geo_df: pd.DataFrame, pat_by_fam: dict) -> dict[str, float]:
    """Per-family Spearman(control within-family geodesic, patristic) — RECOVERY vs the gene tree
    (the matched-human analogue of the cross-kingdom taxonomy recovery)."""
    from scipy.stats import spearmanr as _sp

    rec: dict[str, float] = {}
    for fam, (fg, pat) in pat_by_fam.items():
        present = [g for g in fg if g in geo_df.index]
        if len(present) < 4:
            continue
        idx = [fg.index(g) for g in present]
        patsub = pat[np.ix_(idx, idx)]
        gg = geo_df.loc[present, present].values
        iu = np.triu_indices(len(present), 1)
        c, r = gg[iu], patsub[iu]
        ok = np.isfinite(c) & np.isfinite(r)
        rec[fam] = float(_sp(c[ok], r[ok])[0]) if ok.sum() >= 4 and np.ptp(r[ok]) > 0 else np.nan
    return rec


def _score_human_layer(
    layer: int,
    nat_dir: Path,
    control_stacks: dict[str, np.ndarray],
    genes: list[str],
    fams_arr: np.ndarray,
    fam_order: list[str],
) -> None:
    """Score control-vs-natural preservation at one block for the matched-HUMAN panel and write
    <nat_dir>/controls/.
    """
    nat_geo_df = pd.read_csv(nat_dir / "evo2_human_geodesic_labeled.csv", index_col=0)
    nat_cen_df = pd.read_csv(nat_dir / "evo2_human_centroid_distances.csv", index_col=0)
    family_order = nat_cen_df.index.tolist()
    nat_meta = pd.read_csv(nat_dir / "metadata.csv")
    fam_of = dict(zip(nat_meta["gene"], nat_meta["family"], strict=False))
    pat_by_fam = _human_patristic_by_family(genes, fam_of)

    within_rows, between_rows = [], [{"condition": "natural", "rho_vs_natural_centroid": 1.0}]
    for control, stack in control_stacks.items():
        emb = stack[layer]
        # Same geometry the natural run / write_run_dir builds: angular k-NN geodesic + centroid.
        _, W = find_min_connected_k(emb, k_min=3)
        geo = compute_geodesic(W)
        cen = compute_centroid_geodesic(emb, fams_arr, fam_order)
        ctrl_geo_df = pd.DataFrame(geo, index=genes, columns=genes)
        ctrl_cen_df = pd.DataFrame(cen, index=fam_order, columns=fam_order)

        common = [g for g in nat_geo_df.index if g in ctrl_geo_df.index]
        fams_common = np.array([fam_of.get(g, "NA") for g in common])
        prows, _ = within_preservation_rho(
            nat_geo_df.loc[common, common].values,
            ctrl_geo_df.loc[common, common].values,
            fams_common,
            family_order,
        )
        rec = _patristic_recovery(ctrl_geo_df, pat_by_fam)
        for fam, n, rho in prows:
            within_rows.append(
                {
                    "condition": control,
                    "family": fam,
                    "n_members": n,
                    "rho_geodesic_patristic": rec.get(fam, np.nan),
                    "rho_geodesic_vs_natural": rho,
                }
            )
        brho = between_preservation_rho(
            nat_cen_df.reindex(index=family_order, columns=family_order).values,
            ctrl_cen_df.reindex(index=family_order, columns=family_order).values,
        )
        between_rows.append({"condition": control, "rho_vs_natural_centroid": brho})
        print(
            f"  {control:16} within(mean)={np.mean([r[2] for r in prows]):.3f}  between={brho:.3f}"
        )

    out_dir = nat_dir / "controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_control_table(
        out_dir / "control_within_scores.csv",
        pd.DataFrame(within_rows),
        generator="embed_and_score_controls:ortholog_all_layers",
    )
    write_control_table(
        out_dir / "control_between_scores.csv",
        pd.DataFrame(between_rows),
        generator="embed_and_score_controls:ortholog_all_layers",
    )
    print(f"Saved {out_dir}/control_within_scores.csv + control_between_scores.csv")
    figure_human_controls(out_dir)


def run_human(
    controls: list[str],
    layer: int,
    nat_dir: Path,
    device: str = "cpu",
    force_reembed: bool = False,
    input_mode: str = "genomic",
) -> None:
    """Score within- and between-family control preservation for one human-panel block."""
    stacks, genes, fams_arr, fam_order = _human_setup(controls, device, force_reembed, input_mode)
    print(f"\n=== scoring blocks.{layer} ===")
    _score_human_layer(layer, nat_dir, stacks, genes, fams_arr, fam_order)


def run_human_all_layers(
    controls: list[str],
    nat_dir_for,
    device: str = "cpu",
    force_reembed: bool = False,
    input_mode: str = "genomic",
) -> None:
    """Every-block control preservation for the matched-HUMAN panel."""
    stacks, genes, fams_arr, fam_order = _human_setup(controls, device, force_reembed, input_mode)
    n_layers = next(iter(stacks.values())).shape[0]
    for layer in range(n_layers):
        nat_dir = nat_dir_for(layer)
        if nat_dir is None or not nat_dir.exists():
            print(f"\n########## blocks.{layer}: SKIP (no natural run dir) ##########")
            continue
        print(f"\n########## blocks.{layer} -> {nat_dir} ##########")
        _score_human_layer(layer, nat_dir, stacks, genes, fams_arr, fam_order)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--control", required=True, choices=ALL_CONTROLS + ["all"])
    ap.add_argument(
        "--panel",
        choices=["human"],
        default="human",
        help="matched-human panel — read the sibling control run dirs and score "
        "control-vs-natural preservation (no taxonomy for human paralogs).",
    )
    ap.add_argument(
        "--natural-run",
        default=None,
        help="Natural run dir. Required unless --all-layers resolves it per layer.",
    )
    ap.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Evo2 block index to tap (default: the production EMBED_LAYER). e.g. 15.",
    )
    ap.add_argument(
        "--force-reembed",
        action="store_true",
        help="(--panel human) Re-embed controls even if the cached layer stack exists.",
    )
    ap.add_argument(
        "--figure-only",
        action="store_true",
        help="For --panel human, regenerate the control-preservation figure only "
        "from the existing control_*_scores.csv — no embedding or scoring.",
    )
    ap.add_argument(
        "--all-layers",
        action="store_true",
        help="Score controls at EVERY block, writing each layer's <run>/controls/. "
        "Resolves each layer's natural run dir automatically (ignores --layer/"
        "--natural-run) and reuses the cached layer stacks (no GPU).",
    )
    ap.add_argument(
        "--input",
        choices=["genomic", "cds"],
        default="genomic",
        help="(--panel human) What Evo2 read: 'genomic' transcript span (default) or "
        "'cds' spliced coding sequence (matches the cross-kingdom ortholog input; "
        "controls embed CDS and score against the -cds natural run dirs).",
    )
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = EMBED_LAYER if args.layer is None else f"blocks.{args.layer}"
    print(f"Device: {device} | tap: {layer} | panel: {args.panel}")
    controls = ALL_CONTROLS if args.control == "all" else [args.control]

    # Human controls embed in-process into <nat>/controls/; both axes are pure preservation.
    if args.control == "all":
        controls = HUMAN_CDS_CONTROLS if args.input == "cds" else HUMAN_CONTROLS
    cds_only = {"codon_shuffle", "synonymous_recode"}
    if args.input != "cds" and any(c in cds_only for c in controls):
        sys.exit(
            "codon_shuffle / synonymous_recode require --input cds (they need a reading frame)"
        )
    run_suffix = "-cds" if args.input == "cds" else ""
    if args.all_layers:
        # Sweep layers are nested one-per-block under a single parent run dir
        # (…_evo2-human-panel[-cds]/blocks<L>), so the layer token is the leaf.
        nat_dir_for = lambda L: resolve_latest_run(  # noqa: E731
            f"results/*_evo2-human-panel{run_suffix}/blocks{L}"
        )
        run_human_all_layers(controls, nat_dir_for, device, args.force_reembed, args.input)
        print("\nDone.")
        return
    if not args.natural_run:
        sys.exit("--panel human requires --natural-run")
    if args.figure_only:
        figure_human_controls(Path(args.natural_run) / "controls")
        return
    if args.layer is None:
        sys.exit("--panel human requires --layer (the block the controls are embedded at)")
    run_human(controls, args.layer, Path(args.natural_run), device, args.force_reembed, args.input)
    print("\nDone.")


if __name__ == "__main__":
    main()
