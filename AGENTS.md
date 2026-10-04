# Instructions for agents working in this repository

## Paper data goes in docs/RESEARCH.md

`docs/RESEARCH.md` is NOT the paper; it is the documentation of our research, from which
the paper is written later.

Everything the paper needs about this model is written in `docs/RESEARCH.md`: problem and
claim, related work, benchmark, evaluation protocol, benchmark flaws, data, model, inference,
training, compute, main results, per-category results, statistics, ablations, sanity
controls, error analysis, qualitative examples, post-training (RL / DAgger), deployment,
limitations, ethics and licences, reproducibility, release, decision log. The full required
list is at the top of that file.

- Every new result, training run, analysis or cost goes into the matching section of
  `docs/RESEARCH.md`, with its number, source file and D-number.
- Every decision also gets a D-numbered entry in `docs/LOG.md`.

## Rules

- Rule 1: the 273 UAV-Flow-Eval test tasks are never flown until the user decides; decisions
  use the validation tasks (`docs/PLAN.md`).
- The S3 bucket name stays out of the repo: use `$env:VLA_BUCKET` / `.env` (see `.env.example`).
- This repo holds only the VLA work; the architecture paper lives in its own repository.
