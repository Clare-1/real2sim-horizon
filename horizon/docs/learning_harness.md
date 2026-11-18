# Artifact-backed learning harness

This runner implements the E0 integration check and E1 fixed-controller factorial in the [research plan](../../Agentic_Learning_Harness_路线方案.md). It wraps the existing state expert, β-mixed DAgger, BC trainer and observable geometry estimator. It is not yet a model-driven planner or a physical-robot harness.

## Entry points

```powershell
# From horizon/. Use the verified Python executable described in the runtime report.
$env:PYTHONPATH='src;.venv\Lib\site-packages'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
& 'C:\Users\WYM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/run_harness.py --config configs/harness_e0.yaml --out results/new_e0_run --workers 4
& 'C:\Users\WYM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/run_harness.py --config configs/harness_e1.yaml --out results/new_e1_run --workers 8
& 'C:\Users\WYM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/report_harness.py results/new_e1_run
& 'C:\Users\WYM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/validate_harness_artifacts.py results/new_e1_run
```

For performance experiments, copy source and configuration before launch and point `PYTHONPATH` and the script path at that snapshot. The current E1 batch uses [its frozen protocol](../results/harness_e1_20261026/PLAN.md) and `results/harness_e1_20261026/source/`. Never overwrite historical results when source/configuration changes. CPU fallback uses the separate `.venv-cpu` dependencies and an explicit `device: cpu` configuration; it is a different experimental runtime.

The 8 GB GPU is usable: [actual CUDA/CPU validation](../results/harness_runtime_20261026/README.md). Rollout workers use NumPy inference and single-thread BLAS; only the learner uses Torch/CUDA. Do not infer that the entire simulation moves to the GPU.

## What is implemented

- Explicit scene/task JSON artifacts feed every job through `Job.task_payload`. The E0 producer imports a synthetic upstream MJCF fixture; the E1 producer exports procedural pack scenes. Neither source is called a natural real-world reconstruction.
- Initial dataset D0 and checkpoint π0 are physically shared within each condition/seed. Four arms load the same bytes, verified by hashes.
- S00/S10 freeze policy weights and add no training data; S01/S11 use the current policy for β-mixed DAgger and retrain on cumulative data.
- S10/S11 use the existing geometry estimator, without task-graph or verifier changes. Observed object poses, held-object estimates and commands are separated from scoring labels. The old ideal gripper encoder channel is retained and explicitly reported.
- New data carry the scene/task and twin hashes that generated them. Old data remain unchanged after calibration.
- Exact label budgets truncate retained samples while charging the entire completed rollout batch. Worker errors and unresolved data shortfalls stop execution. This avoids the historical collector's omission of later completed jobs after its budget break.
- Every completed round stores policy, data, observations, geometry estimates, metrics and resource records. JSON writes are atomic. Resume checks data/metadata/model/task references and source/configuration identities; source drift is checked before committing a resumable result.
- Pre-calibration, post-calibration and post-training evaluation use paired developer tasks. Equal system states reuse the same measurement and incur no new rollout cost. Final developer tasks are a separate bank. The reserved confirmation bank is prohibited by this runner.

## Evaluation reuse and training observations (2026-10-28)

`reuse_pre_evaluation: true` additionally reuses the immediately preceding round's post-training evaluation as the next round's pre-calibration evaluation. The key binds the experiment fingerprint (source, configuration, task banks, modes, counts and worker settings), policy bytes and twin belief. A mismatch stops execution. Reused results are deep copies, reference their source round and have zero physical rollout cost; both the independent artifact validator and report loader check this contract. This saves redundant evaluation work, not training data or learning compute. The default remains `false` for compatibility.

`training_observation_mode` is explicit and defaults to `legacy`: noisy object observations with the historical ideal gripper encoder features. `observed` instead builds BC features from the gripper observation already supplied to the controller. Logging does not make extra sensor calls or draw extra noise. Missing, non-finite or incorrectly shaped observed gripper logs fail in this mode rather than silently using ideal encoders. Geometry calibration still uses its separately documented ideal gripper channel. This option changes the training distribution and needs its own controlled performance experiment.

The [three-seed full-H replication plan](../results/long_horizon_replication_20261028/PLAN.md) fixes `legacy` and enables evaluation reuse. Tiny cached/uncached integration runs verified identical dataset/checkpoint hashes and per-episode outcomes; the observed mode also completed integration, without a performance claim. The [integration record](../results/long_horizon_replication_20261028/integration_validation.json) is separate from main experiment evidence.

The [completed replication](../results/long_horizon_replication_20261028/README.md) finished on 2026-10-31: 15 systems, 45 CUDA training calls, 360,000 optimizer steps and 23,670 physical rollouts. All artifacts and 204 regression tests passed. Calibration plus DAgger reached H≥8 SR 75.56±4.59% nominal and 71.94±6.03% perturbed (three-seed mean/sample SD); its gain over calibration with frozen policy was positive in every seed. Healthy DAgger had higher long-H means than recovery data, but seed 2 regressed and family tradeoffs remained. All 1,350 cumulative stage cells were nonempty and none of the 45 trainings used constant-coordinate normalization. The explicit boundary reuse removed 3,600 duplicate rollouts. Raw worker timers span a cross-date interval and must not be interpreted as CPU time or throughput; see the linked runtime note and independent cost audit.

`scripts/report_long_horizon_replication.py` requires all 15 systems and passing independent validation in the three run directories. It revalidates actual artifacts read-only, verifies shared initial states and task pairing, and publishes mean/sample-SD results across three training seeds. It rejects partial matrices and changed inputs. Shared evaluation tasks across seeds are not counted as independent task banks. PNG/SVG export is supported by the verified local runtime; use `report_harness.py --no-plots` for each individual directory when its PDF backend is unavailable.

## Output layout

```text
experiment.json                    source/config fingerprint
tasks/*.json                       explicit scene, graph and task metadata
tasks/*.identity.json              original task content hash
shared/<condition>/s<seed>/         D0, π0, collection and training records
artifacts/<condition>/<arm>_s<seed>/round_<r>/
  feedback/                        tool-visible observation arrays and manifests
  data.npz + data.npz.meta.json     new training shard, when learning is enabled
  policy.npz + policy.train.json   trained policy and actual optimizer statistics
  collection.json                 every completed training rollout and retained labels
  round.json                       version transitions, effects and resource ledger
runs/<condition>/<arm>_s<seed>.json final complete run record
completion.json                    configured grid completed
RESULTS.md / *.csv / *.png / *.pdf validated descriptive reports
```

An uncommitted interrupted round may need to run again. Do not treat the discarded attempt as free performance evidence; keep its logs and partial artifacts. Completed rounds and final runs are reusable only when referenced files still match their original hashes.

## Sources and reuse

| Source | Use in this implementation | Attribution boundary |
|---|---|---|
| Existing HORIZON v1–v4 | State expert, Markov recovery, compositional data, β-DAgger, BC, geometry estimation | The learner and estimator are existing baseline tools, not a new algorithm introduced by this runner. |
| ARSTAG paper recipe recorded in `baselines/BASELINES.md` | Initial recovery mixture and task-consistent randomization retained | An existing paper-inspired reimplementation, not official ARSTAG code or a newly verified full reproduction. |
| SenseNova-RoboRSI evaluation/agent documentation | Separation of runtime/scoring, explicit candidate/version traces, matched evaluation interfaces | Design reference; no upstream model/agent implementation is imported here. |
| Agentic RSR Stage-3 fixture | E0 input→task→expert data→policy learning integration | Synthetic test asset, no natural reconstruction or original robot-controller preservation claim. |
| LiLo-VLA / CycleVLA local references | Existing predicate/recovery context and motivation for long-horizon evaluation | No new official-method score or code reproduction is claimed in this batch. |

Repository references: [baseline inventory](../../baselines/BASELINES.md), [SenseNova agent contract](../../baselines/SenseNova-RoboRSI/docs/agents.md), [SenseNova evaluation contract](../../baselines/SenseNova-RoboRSI/docs/evaluation.md). Broad dependency/model replacement is deliberately deferred until a controlled experiment identifies a relevant limitation.

## Interpreting the results

SR is the fraction of episodes satisfying the fixed full-task truth predicate. Judged completion and partial progress are separate diagnostics. Per-round calibration/learning gains do not replace final paired comparisons, and a positive joint outcome does not by itself establish a positive 2×2 interaction.

E0 and `harness_e1_smoke.yaml` deliberately have tiny datasets/training counts. They verify execution, artifacts and treatment permissions; their success rates are not performance claims. E1 pack H=2/4 is a mechanism development study, not the original multi-family H=2..10 scaling experiment. The model planner, cross-task improver transfer, natural reconstruction and independent-target deployment remain separate subsequent stages.

## Completed E1 and separate development diagnostics

The [2026-10-26 E1 matrix](../results/harness_e1_20261026/RESULTS.md) has 24 complete runs and 72 complete rounds. Both report validation and the independent artifact validator passed. [Core regression tests](../results/harness_validation_20261026/README.md) passed 109/109. The GPU executed 42 actual training calls; shared initial training is counted once in `artifact_validation.json`, which also records physical rollout costs.

The existing estimator failed on one frozen-policy trajectory: repeated oscillation windows dominated frame-weighted support estimation and changed an approximately 20 mm belief to 94 mm. The [diagnosis](../results/harness_e1_20261026/GEOMETRY_DIAGNOSTIC.md) preserves this run and precisely reproduces its saved observations. No threshold or algorithm was changed in the frozen matrix.

Two follow-up evaluators keep their own plans and outputs:

- `evaluate_harness_first_calibration.py` compares the initial and final S11 checkpoints at the identical **first-round** calibrated belief on the already observed H2/4 final development bank. It is post-hoc, does not train another policy, and reuses an evaluation only when policy bytes, belief, config and task identities match. In this batch 11 system evaluations were reused and 80 new rollouts were executed. [Results](../results/harness_first_calibration_20261026/DIAGNOSTIC.md).
- `evaluate_harness_transfer.py` evaluates all final E1 systems at H6/8/10, without training/calibration, on the new 68M development bank. All 24 systems and 2,880 rollouts completed with zero full-task successes; [results and limits](../results/harness_transfer_20261026/TRANSFER.md). This is extrapolation from H2/4 training, not a full-range learning scaling study. The separate `check_harness_transfer_expert.py` only checks 6 fixed task identities with the privileged expert; 12/12 sampled episodes succeeded. [Expert check](../results/harness_transfer_expert_20261026/README.md).

These scripts require the complete E1 matrix and its frozen source:

```powershell
$env:PYTHONPATH='results\harness_e1_20261026\source\src;.venv-cpu\Lib\site-packages'
& 'C:\Users\WYM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/evaluate_harness_transfer.py --source-run results/harness_e1_20261026 --out results/harness_transfer_20261026 --workers 8
& 'C:\Users\WYM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/evaluate_harness_first_calibration.py --source-run results/harness_e1_20261026 --out results/harness_first_calibration_20261026 --workers 2
```

Completed system records are verified and resumed; source/model/task/config or evaluator changes require a new output directory. The first-calibration evaluator also hashes its `evaluate_harness_transfer.py` helper. CPU dependencies suffice for these evaluations because saved policies run through NumPy.

[RoboDojo](robodojo_appendix.md) remains optional. Native Score, full-task SR and paired sim-real correlation are distinct; its current agent interface and Windows/8GB feasibility have not been verified. No leaderboard submission is part of these experiments.

## Long-horizon development extensions (2026-10-28)

New settings are opt-in; historical configuration defaults remain unchanged:

- `calibration_schedule: first_round` permits S10/S11 to estimate geometry only in round one and freezes the resulting belief for later rounds, including when the first estimate changes nothing. `every_round` is the legacy default. Feedback is still collected and charged each round for equal opportunities; scores never decide whether to accept a calibration.
- `learning_collection: nominal_recovery` adds fresh successful recovery-expert demonstrations as the matched retained-label/optimizer-budget control. `dagger` remains the default. Failed/overshoot expert rollouts are charged; DAgger may retain labels from unsuccessful episodes. These are different physical collection costs, not a claim of equal elapsed time.
- `train.normalization: constant_safe` uses unit scale and the exact first observed value for exactly constant training coordinates. All varying dimensions retain the original mean and `(std + 0.001)`. It does not mask obstacles, remove H/k, or learn statistics from evaluation. `legacy` retains the original numerical behavior; saved policies remain compatible.
- Evaluation records now retain terminal satisfied-predicate count, maximum simultaneous count including the last state, and first unsatisfied terminal index. These scoring diagnostics do not enter feedback. `family_horizon_metrics.csv` separates task families and H; terminal prefix completion is not a history of reaching stages.

The three-seed [same-checkpoint scale intervention](../results/horizon_normalization_20261028/RESULTS.md) completed 600 rollouts. Changing only the absent fourth-neighbour scales did **not** improve full SR; all long-H outcomes remained zero. The numerical hazard is real, but it is not a sufficient explanation or a validated solution to short-H extrapolation.

The [completed development screen](../results/long_horizon_dev_20261028/RESULTS.md) restores all three families and H2/4/6/8/10 in training. Its one-seed five-system results are separate from the short-H diagnostic and from the subsequent three-seed development replication; neither opens the reserved confirmation bank. [Local baseline design](long_horizon_baseline_design.md) records the exact sources and reuse boundaries; no upstream model is represented as fully reproduced.

Two implementation questions remain separate from the frozen learning comparison. Training defaults to ideal gripper features while deployment perception adds 0.5 mm noise; the new observed-input option above supports a future controlled ablation. Mixed-controller termination follows the learner's selected subgoal even on expert-executed steps. A [90-rollout diagnostic](../results/mixed_termination_20261028/README.md) found zero strictly defined false-done terminations on its sampled tasks/checkpoints, so the replication retains the existing controller. Neither question has been established as the cause of the earlier short-H extrapolation failure.
