# Formalize Once, Solve Many

Code and run records for *"Formalize Once, Solve Many: When to Write Code Instead of
Reasoning Per Case on Rule-Governed Tasks"* (COLM 2026 Workshop on Efficient Reasoning).

A language model facing a rule-governed task can **reason through** each case, or
**formalize** the rules once into a program and run that program on every case. This
repository measures both strategies on [RuleArena](https://github.com/skyriver-2000/RuleArena)
— airline baggage fees and U.S. Form 1040 tax — across five frontier models and three
difficulty tiers.

The headline: formalize-once costs 10–50× less, because one program amortizes over the
whole test set, and it is all but never less accurate. Across the 35 cells we report,
per-case reasoning wins significantly in 1 and formalize-once in 14.

## Reproducing the paper's numbers

Every number in the paper is recomputed from the run records in
`data/results/rulearena/`. **No API key, no network access and no spend are needed** —
the generators read the stored per-case outcome vectors and never call a model:

```bash
python reproduce.py            # regenerate all macros, tables and the figure
python reproduce.py --check    # and run the test suite
```

That writes `paper/ra_*.tex`, three tables and `paper/figures/ra_forest.pdf`, then
prints the headline figures for comparison against the paper. A disagreement means the
paper is wrong, not the code.

Nothing else in this repository needs to be installed or run first. `reproduce.py` is
cwd-independent and idempotent.

## Spending money is opt-in

Reproduction is free, and the code enforces that rather than merely documenting it.
Every paid call in the project passes through one function, which refuses to fire
unless you opt in:

```
RuntimeError: refusing to make a paid API call for gpt55: this call is not in the
local cache, so completing it would cost money.
```

To re-run an experiment against live endpoints, set `OPENROUTER_API_KEY` (see
`.env.example`) **and** `SSCRAMBLE_ALLOW_SPEND=1`, then use the runners in `tools/` —
`run_sweep.py`, `run_fresh.py`, `run_extract.py`, `run_taxl1_strat.py`. Price a run
first with `tools/preflight.py`. Calls already in the cache replay for free and do not
need the opt-in, so an interrupted run resumes without re-buying anything.

The call cache itself (`data/results/rulearena/callcache/`, 26 MB) is not included in
this export, so on a fresh clone every live call is a cache miss.

## Reading the run records

Each record in `data/results/rulearena/*.jsonl` carries its own per-case outcomes, so
analysis needs no call cache. Two fields matter if you write your own analysis, because
both exclude cases from the paired test:

- `per_case_no_answer` — the model spent its whole completion ceiling reasoning and
  returned no text. A real model outcome, but not a wrong answer, so it is excluded
  rather than scored incorrect.
- `per_case_plain_err` — the call never completed (an infrastructure failure). **A row
  with any of these is an abandoned run and must be dropped entirely.** One such row is
  present and superseded: a `claude_opus48` row in `airline_fresh_core.jsonl` where 87
  of 90 plain calls errored. Taken at face value it reports a `+0.867` gap; the clean
  rerun beside it reports `+0.078`. `tools/ra_cells.py` is the single source of truth
  for which rows and cases count.

Superseded and withdrawn runs are kept in place with an explanatory suffix
(`.WITHDRAWN`, `.BADENVELOPE`, `.PLACEHOLDER`, `.UNMATCHEDDIST`, …) so the corrections
are inspectable. No script reads them. The largest is worth naming: an early
contamination-control run put Opus at `+0.356`; the corrected run that replaced it says
`+0.078`.

## Layout

```
reproduce.py                     one offline entry point; runs the 11 generators below
tools/fill_ra_*.py               each writes one group of paper macros from the records
tools/cost_curve.py              break-even and the cost curve
tools/loss_report.py             which cases each strategy trades
tools/ra_cells.py                which rows and cases are in scope  <- read this first
tools/ra_stats.py                paired McNemar exact test
tools/run_*.py                   live runners (need SSCRAMBLE_ALLOW_SPEND=1)
tools/preflight.py               price a run before you start it
src/symbol_scramble/rulearena/   the experiment: formalize, execute, grade, cache
src/symbol_scramble/llm/         model clients; the spend guard lives here
data/results/rulearena/          run records, including superseded ones
data/raw/rulearena/              vendored RuleArena rulesets and fee tables
configs/models.yaml              model ids and the per-token prices used for costing
tests/                           28 tests on a clone, none of which need network
```

The Python package is still named `symbol-scramble` in `pyproject.toml`: this study
grew out of an earlier project and the import path was left alone rather than churned
after the results were final.

## Citation

```bibtex
@inproceedings{subramanian2026formalize,
  title     = {Formalize Once, Solve Many: When to Write Code Instead of
               Reasoning Per Case on Rule-Governed Tasks},
  author    = {Subramanian, Sathvik},
  booktitle = {COLM 2026 Workshop on Efficient Reasoning},
  year      = {2026}
}
```

RuleArena is vendored under `data/raw/rulearena/` and retains its own license; see
`NOTICE.md`. Everything else is MIT (`LICENSE`).
