# Minimal validation: selector-driven learning-operation choice

Date: 2026-11-12. Implements the advisor's §三.6 gate before any large experiment.

## What this run answers

Each scenario hides one failure cause. Every round the policy is evaluated on the **full-support**
deployment distribution, the deployment feedback is compressed into a leak-proof `FailureProfile`,
and a selector picks one learning operation. The primary metric is **whether round one picked the
operation that helps** — a binary with an objective answer, unlike SR at this sample size.

| scenario | hidden cause | correct first operation | what the twin does wrong |
|---|---|---|---|
| `geometry` | `geometry_bias` | `calibrate_geometry` | support height believed 6 mm vs 20 mm |
| `coverage` | `coverage_deficit` | `regenerate_targeted` | nothing — the data only covers x ≤ 0 |

## Files

- `run/` — the reduced-budget formal run (`configs/minimal_validation.yaml`).
- `coverage_deficit_acceptance.log` — piece-1 acceptance (expert 10/10 on deployment tasks).
- `pytest_full.log` — full regression at the time of the run.

## Pipeline smoke (tiny budget, 6/6 runs complete)

Config: `configs/minimal_validation_smoke.yaml` (300 init labels, 4 feedback episodes, 20 train steps).
**SR is meaningless at this scale (all 0.000); only the action chain is informative.**

| selector | first step correct (geometry) | first step correct (coverage) |
|---|---|---|
| fixed  | 1/1 | **0/1** |
| rule   | **0/1** | 1/1 |
| oracle | 1/1 | 1/1 |

Decision chains:

```
geometry  fixed   s0: calibrate_geometry -> dagger_round
geometry  rule    s0: regenerate_targeted -> regenerate_targeted
geometry  oracle  s0: calibrate_geometry -> calibrate_geometry
coverage  fixed   s0: calibrate_geometry -> dagger_round
coverage  rule    s0: regenerate_targeted -> regenerate_targeted
coverage  oracle  s0: regenerate_targeted -> regenerate_targeted
```

Two things this already shows:

1. **The fixed recipe fails exactly where the hypothesis says it should.** On coverage deficit it
   spends its first move calibrating a twin that is already correct, and never touches the
   uncovered region. It is correct on geometry. This is the "recipe–cause mismatch" the paper
   needs, reproduced in the smallest possible setting.
2. **The rule planner misdiagnosed geometry at 4 feedback episodes.** With too few episodes
   `geometry_repair` returns no usable residual, so the residual branch never fires and the rule
   falls through to the concentration branch. This is not a bug in the rule; it is the
   identifiability term ε in the advisor's proposition 2 — diagnosis costs feedback, and below a
   threshold the selector is blind. The formal run uses 24 feedback episodes for this reason.

**Neither point is an end-to-end result.** With 300 initial labels and 20 optimisation steps every
policy scores 0, so nothing here says which operation improves the policy. It says the decision
layer behaves as designed.

## Final 18-run result (complete) — three assumptions refuted

| selector | first step correct (coverage) | first step correct (geometry) | final SR (coverage) | final SR (geometry) | new labels |
|---|---|---:|---:|---:|---:|
| fixed  | **0/3** | 3/3 | **0.083** | 0.042 | 24000 |
| rule   | 2/3 | 3/3 | **0.000** | **0.208** | 18000 |
| oracle | 3/3 | 3/3 | **0.000** | **0.000** | 18000 |

**This run is archived as evidence that three design assumptions were wrong. It is not a result
about the method and must not be cited as one.**

1. **The oracle was not an upper bound.** It scored 0.000 everywhere and lost to both baselines,
   because a single-action oracle calibrates on every round and never trains. Fixed by making the
   oracle follow a per-cause *sequence* (`ORACLE_SEQUENCE`).
2. **The hard-coded cause→action map did not predict improvement.** On coverage the rule picked
   the "correct" first step and scored 0.000, while the fixed recipe picked the "wrong" one and
   scored 0.083. The map must be measured (R1 matrix), not asserted.
3. **The action names were semantically inverted.** `regenerate_targeted` sampled from the *training*
   prior and `regenerate_uniform` from the *deployment* store — the opposite of their names. Fixed:
   targeted now resamples where deployment actually puts objects, uniform resamples the training prior.

A fourth, found while calibrating: **the reduced budget is itself broken.** At 20k labels / 4000
steps the policy scores 0.000 even with a *correct* twin (bias factor 1.0), so no bias level could
have produced a usable curve. Calibration and R1 must run at the protocol budget (80k / 8000).


`configs/minimal_validation.yaml`: 20k init + 3×12k labels, 24 feedback episodes, 4000 train steps,
pack H=4, 2 causes × 3 training seeds × 3 selectors = 18 runs.

This is deliberately smaller than the 80k + 3×40k recipe in the specification so the screening
finishes in one session. **The SR column is directional only**; the primary metric is unaffected.

**Status at time of writing: running in the background.** First cells show `SR 0.000 -> 0.000`,
which is expected at 20k labels / 4000 steps — the policy is under-trained, so the SR columns of
this screening run will be uninformative. If they stay pinned at zero, the honest conclusion is
that the screening budget is too small to say anything about SR, and only the first-step column
should be read. The formal table in the specification (80k + 3×40k) is what the paper would need.

### Partial result: the ε effect confirmed

With the smoke's 4 feedback episodes the rule misdiagnosed geometry. With the formal run's 24
episodes it now picks correctly:

```
geometry  fixed  : calibrate_geometry -> dagger_round -> dagger_round
geometry  rule   : calibrate_geometry -> calibrate_geometry -> ...
```

So the earlier misdiagnosis was **feedback volume, not rule logic** — exactly the identifiability
term ε in proposition 2. Practical consequence for the paper: the diagnosis step has a minimum
feedback cost, and any claim of the form "the agent diagnoses the cause" must report how many
episodes that took.

SR remains `0.000 -> 0.000` in both, which is the budget problem above, not a selector problem.



## Reproduction

```bash
cd horizon
PYTHONPATH=src .venv-linux/bin/python scripts/run_minimal_validation.py \
    --config configs/minimal_validation.yaml \
    --out results/minimal_validation_20261112/run --workers 8
PYTHONPATH=src .venv-linux/bin/python scripts/report_minimal_validation.py \
    results/minimal_validation_20261112/run
```
