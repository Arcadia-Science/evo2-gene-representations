"""Download resumable Ensembl GTF, genome FASTA, and validation CDS files for the mammal panel."""

from __future__ import annotations
import gzip
import json
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
from resolve_orthologs import SPECIES  # noqa: E402  the 24-species panel

REL = "release-116"  # MUST match the release the REST Compara orthology used (info/data -> 116)
BASE = f"https://ftp.ensembl.org/pub/{REL}"
# ~3.3 GB of Ensembl FASTA + GTF. Override with $GLM_MAMMAL_GENOMES or $GLM_SCRATCH to reuse a
# copy that already exists rather than downloading it again.
DEST = paths.MAMMAL_GENOMES


def _listing(url: str, tries: int = 4) -> str:
    for a in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=30).read().decode()
        except Exception:
            time.sleep(2**a)
    return ""


def _pick(listing: str, suffix: str) -> str | None:
    hits = [h for h in re.findall(r'href="([^"]+)"', listing) if h.endswith(suffix)]
    return hits[0] if hits else None


def _download(url: str, dest: Path, decompress_gz: bool = False) -> None:
    """Stream url -> dest, resumable-by-existence. If decompress_gz, gunzip while writing."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"    have {dest.name}", flush=True)
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    t = time.time()
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                if decompress_gz:
                    with gzip.GzipFile(fileobj=r) as gz, open(tmp, "wb") as out:
                        shutil.copyfileobj(gz, out, length=1 << 22)
                else:
                    with open(tmp, "wb") as out:
                        shutil.copyfileobj(r, out, length=1 << 22)
            tmp.replace(dest)
            print(
                f"    {dest.name}  ({dest.stat().st_size / 1e6:.0f} MB, {time.time() - t:.0f}s)",
                flush=True,
            )
            return
        except Exception as e:
            print(f"    retry {dest.name} ({type(e).__name__})", flush=True)
            time.sleep(2**attempt)
    print(f"    FAILED {dest.name}", flush=True)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    prov = {"release": REL, "species": {}}
    for i, sp in enumerate(SPECIES, 1):
        d = DEST / sp
        d.mkdir(exist_ok=True)
        print(f"[{i}/{len(SPECIES)}] {sp}", flush=True)
        gtf_ls = _listing(f"{BASE}/gtf/{sp}/")
        gtf = _pick(gtf_ls, f".{REL.split('-')[1]}.gtf.gz")
        dna_ls = _listing(f"{BASE}/fasta/{sp}/dna/")
        dna = _pick(dna_ls, "dna.toplevel.fa.gz")
        cds_ls = _listing(f"{BASE}/fasta/{sp}/cds/")
        cds = _pick(cds_ls, "cds.all.fa.gz")
        if not (gtf and dna):  # CDS optional (validation only)
            print(f"    MISSING required listing gtf={bool(gtf)} dna={bool(dna)}", flush=True)
            continue
        # assembly token = middle of the filename: <Species>.<assembly>.dna.toplevel.fa.gz
        asm = dna.replace(".dna.toplevel.fa.gz", "").split(".", 1)[1]
        _download(f"{BASE}/gtf/{sp}/{gtf}", d / gtf)  # required
        _download(f"{BASE}/fasta/{sp}/dna/{dna}", d / dna[:-3], decompress_gz=True)  # required
        if cds:
            _download(f"{BASE}/fasta/{sp}/cds/{cds}", d / cds)  # validation only
        prov["species"][sp] = {"assembly": asm, "gtf": gtf, "dna": dna[:-3], "cds": cds}
        (DEST / "provenance.json").write_text(json.dumps(prov, indent=2))
    print("DOWNLOADS DONE", flush=True)


if __name__ == "__main__":
    main()
