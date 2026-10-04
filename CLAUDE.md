# CLAUDE.md

Read `AGENTS.md` first; it applies here in full.

## Paper data goes in docs/RESEARCH.md

All data the paper needs is written in `docs/RESEARCH.md` (required list at its top: problem
and claim, related work, benchmark, evaluation protocol, benchmark flaws, data, model,
inference, training, compute, main results, per-category results, statistics, ablations,
sanity controls, error analysis, qualitative examples, post-training, deployment,
limitations, ethics and licences, reproducibility, release, decision log).

- After every result, run, analysis or cost: update the matching section of
  `docs/RESEARCH.md` (number, source file, D-number) and add the decision to `docs/LOG.md`.
- Rule 1: never fly the 273 test tasks; decide on validation.
- Never write the S3 bucket name into the repo; use `$env:VLA_BUCKET`.
