# Full-H three-seed replication and independent implementation diagnostics

Fixed before this batch's performance outcomes, 2026-10-28. This follows the prior single-seed full-H development screen and the prepared three-seed configurations. Development replication, not a reserved confirmation experiment or agent decision test.

## Main replication (15 final systems, 45 CUDA training calls)

- Three training seeds: 0, 1, 2. Pack/sort/stack × H2/4/6/8/10 in both training and evaluation. Original 80k initial labels + three 40k increments; 8000 optimization steps for each trained model; same expert, MLP, β=0.5^round, recovery recipe, environment, judge and task graph as the preceding screen.
- Healthy: matched retained-label/optimizer-budget recovery-expert additions vs standard DAgger. Within each seed verify the two directories' initial data and weights are identical. Count duplicated physical initial training and generation separately.
- Height bias: fixed twin + DAgger (S01), first calibration + frozen policy (S10), first calibration + DAgger (S11). Share D0/π0 and first feedback within each seed. Freeze the first belief regardless of its quality; do not select calibration rounds or delete bad seeds. Frozen policy is a lower-training-cost mechanism control.
- New banks: training84M, feedback85M, curve86M, final87M; reserved40M is untouched. Feedback30/round, curve4 and final20 tasks per family×H per mode. All methods and seeds share final task identities; 300 unique final identities, not 900 independent trained systems. Nominal and p=.15 perturbation modes remain separate.
- Report seed mean ± sample SD, each seed's result, paired per-seed differences, family×H, H≥8, terminal prefix and real training-stage coverage. No significance/SOTA claims or full-factorial interaction without S00.
- Primary questions: can long-H performance repeat across seeds; do calibration and subsequent learning retain value; is DAgger's family-specific tradeoff reproducible? Report the entire matrix even if the answer is negative.

## Implementation changes kept separate from the learning treatment

- `reuse_pre_evaluation: true` reuses only the immediately preceding post-training measurement, after checking identical experiment/source/config fingerprint, policy bytes and belief. Task references remain hashed. The evaluation bank/modes/counts are bound by the experiment identity. There are no model/geometry changes between these two measurements. Count each reused pre-evaluation as zero physical rollouts; do not count the saved evaluation as a learning-efficiency improvement. This saves exactly 15 systems × 2 later rounds × 120 rollouts = 3600 redundant evaluations if every run completes.
- `training_observation_mode: legacy` in every main run. The new observed-gripper training option is **not** used in this replication. Tests verify default arrays/actions/random observation draws match the old implementation. Original ideal encoder feedback remains intact.
- CUDA learner and CPU simulation remain separate. Healthy runners use four workers each; the longer bias runner uses twelve (maximum twenty rollout workers across the three runners). Retained label sequences are ordered and job seeds are fixed. Actual batch overshoot costs are charged; concurrent wall times are not a controlled speed benchmark.

## Integration and independent diagnostics

Before main training: full tests, then tiny two-round cached/uncached legacy runs and an observed-input run. Cached and uncached legacy runs must have identical policy/data/episode results; independent validators must accept zero-cost reuse and the explicit input channel. Four optimization steps per smoke call are integration only.

The separate `mixed_termination_20261028` diagnostic uses old frozen source/checkpoints and a new88M task bank. Its ninety rollouts do not train or change the controller. Do not modify MixedController based solely on a theoretical risk.

All performance runs execute frozen code and configs in separate directories. Report source/code/artifact fingerprints, preserve partial work and failures, and never mutate historical results. A follow-up observed-input or coverage-weight experiment must have its own frozen plan and controls; neither is silently pooled with the main replication.
