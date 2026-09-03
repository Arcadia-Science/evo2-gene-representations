# Repository refactoring plan

This file is the source of truth for the staged repository refactor agreed with the repository
owner. Before beginning a stage, review the relevant code read-only, propose a detailed plan, and
wait for explicit approval. After completing it, summarize changed files, active-code LOC added and
removed, verification results, and any remaining risks before proceeding.

## Working rules

- Consolidate and archive; do not delete repository history or functionality.
- Code retired from the active repository moves to `deprecated/` with documentation of its former
  path, purpose, restoration procedure, and dependencies.
- Prioritize removing and reusing existing code. Do not introduce new modules, abstractions, or
  algorithms unless correctness makes a small addition unavoidable and it is approved first.
- Preserve user changes and keep `results/` untouched unless a later stage explicitly authorizes a
  particular result regeneration.
- Complete one stage at a time: detailed proposal, approval, implementation, verification, and
  summary.

## Prioritized refactoring plan

1. **Freeze current outputs and resolve scientific mismatches.** Determine whether Figures 1 and 2
   should use W2/angular or the current geodesic paths. Add characterization tests before code
   movement. High value; no refactor is safe without it.
2. **Fix unstable random seeding and make CI real.** Replace string `__hash__()` seeding, add the
   core numerical/control fixtures, remove `--exit-zero`, and run pytest. Very high value, low risk.
3. **Delete definite dead code and dead functions.** Remove `control_ladder.py`,
   `within_family_permutation.py`, dormant extraction code, and uncalled helpers. Low risk.
4. **Centralize configuration, FASTA I/O, family metadata, and control definitions.** This
   eliminates widespread small duplication without changing algorithms. Low to medium risk.
5. **Collapse the Arcadia style copies and thin plotting wrappers.** Large LOC/file-count reduction
   with contained visual risk.
6. **Consolidate mammalian dataset construction.** Remove inactive dataset tracks and merge
   cap/mask/manifest stages. Medium risk.
7. **Prune figure generators to published panels.** Preserve golden source tables and visual checks
   while removing diagnostic branches. Medium risk.
8. **Unify family geometry and control scoring around the scientifically selected metrics.** Remove
   unused FGW/general OT and parallel graph scorers. High scientific value and high risk.
9. **Merge direction geometry, gates, and evolutionary predictor-table construction.**
   Medium-to-high risk; strongly reduces Experiment 3 conceptual duplication.
10. **Unify nucleotide site definitions, alignment, rescoring, and directionality.** High value but
    high risk because it affects primary steering outcome metrics.
11. **Unify ortholog acquisition and transcript policy.** Do this only after a full
    canonical-versus-longest-CDS comparison. High risk.
12. **Move Figure 6 to the 400-gene panel and remove the old paired-panel route.** Potentially large
    simplification, but highest scientific risk and requiring explicit approval.

## Status

- Phase 1: complete. Publication paths use uncapped W2 between families and angular distance within
  families; legacy figures and code are recoverable.
- Phase 2: complete. Future control generation is deterministic, local/CI checks fail correctly,
  and the active tests are recovered archived fixtures rather than newly authored test logic.
- Phase 3: complete. Dormant entry points are archived, live locus helpers are consolidated into
  dataset assembly, and geodesic-only within-family inference is retired from active code.
- Phase 4: complete. Twelve FASTA readers, the duplicate mammalian CDS loader, control membership,
  control-table ownership, and ortholog family access are consolidated into existing modules. The
  sole generic table wrapper is recoverably archived. Active Python changed by +138/-308 lines
  (net -170); 66 tests and Experiment 1–3 capability checks pass without touching outputs.
- Phase 5: next. Perform a read-only review and obtain approval for a detailed plan before editing.
- Phases 6–12: pending.

## Detailed audit specification

The complete original architecture audit—including component data flows, current files, rationale,
proposed destinations, archive candidates, risks, protective tests, duplication inventory, CI
findings, and dependency notes—is preserved verbatim in
[`REFACTORING_PLAN_DETAILS.md`](REFACTORING_PLAN_DETAILS.md). It is a required companion to this
plan and must be consulted during every phase review.

The detailed audit sections map to the prioritized phases as follows:

| Phase | Detailed audit material |
|---|---|
| 1 | “Important data-flow discrepancies” items 1–3 and the Figure 12 metric investigation |
| 2 | Data-flow discrepancy 4, “Tests and CI,” and the control/numerical fixture requirements |
| 3 | Consolidation recommendation 1 and the dead/thin-wrapper inventory |
| 4 | Consolidation recommendations 2, 5, and 10 |
| 5 | Consolidation recommendation 3 |
| 6 | Consolidation recommendation 4 |
| 7 | Consolidation recommendation 7 |
| 8 | Consolidation recommendation 6 |
| 9 | Consolidation recommendation 9 |
| 10 | Consolidation recommendation 8 |
| 11 | Consolidation recommendation 11 |
| 12 | Consolidation recommendation 12 |

## Owner decisions overriding original audit wording

- Every recommendation that says “delete” means **move to `deprecated/` with documentation of the
  former path, purpose, dependencies, restoration procedure, and verification**.
- Proposed new modules such as `controls.py`, `figure_style.py`, `family_geometry.py`, or
  `experiment*_figures.py` describe conceptual ownership only. Prefer consolidating into existing
  files and reducing active file count; creating a new module requires explicit approval.
- Do not replace `gene_families.py` with a JSON-only loader or obscure how `families_data.json` was
  generated. Its transparent generation path must remain available to users.
- Keep optional Wasserstein Mantel capability available, although it is not part of the default
  publication pipeline. Retire geodesic-only within-family Mantel code in Phase 3.
- Keep paired-p3 available as a separate experiment and Figure 12 source; do not silently add it to
  the ordinary Experiment 2 control list.
- Keep `results/` read-only unless a later phase explicitly authorizes a named regeneration.
- Reconsider Figure 6 on the 400-gene panel only in Phase 12, with an explicit old/new scientific
  comparison before archiving the older route.
