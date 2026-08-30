"""Diagnostic-site definition on the NUCLEOTIDE alignment — the ground truth for a DNA model."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts" / "steering"))

from top_recon_steer_sweep import nt_aligner  # noqa: E402

BASES = frozenset("ACGT")


def diagnostic_sites_nt(human_window: str, target_window: str) -> dict[int, str]:
    """{index into `target_window` -> platypus base} where the nucleotide alignment pairs that platypus base with a DIFFERENT human base."""
    if not human_window or not target_window:
        return {}
    a = nt_aligner().align(human_window, target_window)[0]
    h_aln, t_aln = str(a[0]), str(a[1])
    out: dict[int, str] = {}
    tcur = 0
    for hc, tc in zip(h_aln, t_aln):
        if tc == "-":
            continue
        if hc != "-" and hc in BASES and tc in BASES and hc != tc:
            out[tcur] = tc
        tcur += 1
    return out


def site_pairs_nt(human_window: str, target_window: str) -> dict[int, tuple[str, str]]:
    """{index into `target_window` -> (human base, platypus base)} for the SAME sites as `diagnostic_sites_nt`."""
    if not human_window or not target_window:
        return {}
    a = nt_aligner().align(human_window, target_window)[0]
    h_aln, t_aln = str(a[0]), str(a[1])
    out: dict[int, tuple[str, str]] = {}
    tcur = 0
    for hc, tc in zip(h_aln, t_aln):
        if tc == "-":
            continue
        if hc != "-" and hc in BASES and tc in BASES and hc != tc:
            out[tcur] = (hc, tc)
        tcur += 1
    return out
