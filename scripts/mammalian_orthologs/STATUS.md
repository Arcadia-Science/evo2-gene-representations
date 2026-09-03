# Mammalian ortholog pipeline status

**Current status:** complete for experiment 1 and for the composition-controls analysis.

The production panel starts from 48 HGNC gene families and resolves one-to-one orthologs across 24
mammals with Ensembl Compara release 116. The assembled manifest contains 11,288 loci. Transcript
spans, CDS sequences, CDS-position masks, and the independent species-tree matrix are stored under
`data/mammalian_orthologs/`.

## Active workflow

| Stage | Entry point |
|---|---|
| Ortholog resolution | `resolve_orthologs.py` |
| Bulk Ensembl annotation and genome download | `download_bulk.py` |
| Local locus extraction and QC | `extract_loci_bulk.py` |
| Manifest assembly and 400-cap sensitivity panel | `assemble_datasets.py`, `build_capped_manifest.py` |
| Independent mammalian tree | `build_species_tree.py` |
| CDS-position masks and all-block embedding | `build_cds_masks_mammal.py`, `embed_cds_masked_mammal.py` |
| Within- and between-family scoring | `mammal_score.py`, `mammal_between.py` |
| Inference | `within_family_uncertainty.py` |
| Composition controls (analyses/controls/) | `mammal_controls_score.py`, `controls_score_graphfree.py` |

The complete command order is maintained in
[`experiments/exp1_gene_family_geometry.sh`](../../experiments/exp1_gene_family_geometry.sh); the
control stages are driven by
[`analyses/controls/run_composition_controls.sh`](../../analyses/controls/run_composition_controls.sh).

Early extraction prototypes used Ensembl REST and were vulnerable to service outages. The active
extractor uses downloaded GTF and genome files for sequence extraction; REST is retained only where
the orthology-resolution stage requires it.
