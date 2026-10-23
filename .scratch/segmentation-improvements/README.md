# Local segmentation improvement tracker

Destination: local files only. Do not publish or synchronize these tickets to GitHub or another external tracker.

Source: the approved segmentation improvement TODO and ticket breakdown. All tickets start with the `ready-for-agent` triage status; this does not override blockers, missing inputs or experiment launch authorization. Work only on tickets whose prerequisites are satisfied.

Initial dependency-free tickets: **01, 02, 05, 07, 13, 17, 21**. Artifact access, hardware and model availability may still gate execution.

Ticket 15 needs 13 only for an organ-aware rule. Ticket 18 requires an explicit selection of pilot tickets for this research round; record their identifiers as blockers before starting it. Do not require every optional pilot or choose the pilot set silently.

| Ticket | Blocked by |
|---|---|
| [01: Recover and inventory the completed B experiment](issues/01-recover-b-experiment.md) | None |
| [02: Make segmentation evaluation reusable and verified](issues/02-verified-reusable-evaluation.md) | None |
| [03: Certify the completed B baseline](issues/03-certify-b-baseline.md) | 01, 02 |
| [04: Review learning curves and inner-validation errors](issues/04-learning-curve-review.md) | 01 |
| [05: Benchmark feasible training patch sizes](issues/05-patch-size-benchmark.md) | None |
| [06: Evaluate a larger-patch training pilot](issues/06-larger-patch-pilot.md) | 02, 05 |
| [07: Audit training-patch coverage](issues/07-sampling-coverage-audit.md) | None |
| [08: Evaluate lesion-balanced sampling](issues/08-lesion-balanced-sampling.md) | 02, 07 |
| [09: Evaluate hard-negative sampling](issues/09-hard-negative-sampling.md) | 02, 07 |
| [10: Evaluate multiple patches per examination](issues/10-multiple-patches.md) | 02, 07 |
| [11: Evaluate a revised learning-rate or stopping policy](issues/11-schedule-stopping-pilot.md) | 02, 04 |
| [12: Evaluate Dice plus cross-entropy loss](issues/12-dice-cross-entropy.md) | 02 |
| [13: Prepare and validate anatomical organ information](issues/13-validated-organ-information.md) | None |
| [14: Evaluate one anatomy-aware model variant](issues/14-anatomy-aware-pilot.md) | 02, 13 |
| [15: Evaluate conservative postprocessing](issues/15-conservative-postprocessing.md) | 02; 13 if organ-aware |
| [16: Establish an nnU-Net comparator pilot](issues/16-nnunet-comparator.md) | 02 |
| [17: Define a lesion-negative cohort experiment](issues/17-negative-cohort-protocol.md) | None |
| [18: Select and freeze the next scientific recipe](issues/18-freeze-scientific-recipe.md) | 03, 04; selected pilots (must be recorded) |
| [19: Complete and verify the frozen five-fold experiment](issues/19-frozen-five-fold-evaluation.md) | 18 |
| [20: Measure training variability with additional seeds](issues/20-training-seed-variability.md) | 19 |
| [21: Design an independent pretrained-model comparison](issues/21-independent-comparison-protocol.md) | None |

**Shared research constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.

GenAI is outside this backlog until its role, data, comparator and measurable endpoint are agreed with Igor.

Implementation update, 2026-10-25: ticket 02 tooling is implemented with CPU
verification and both 145-examination patch evaluations are complete. Ticket 07
is complete as a bounded six-examination training-only audit, including a verified
comparison. Ticket 08 is implemented/CPU-tested with a frozen paired pilot and
inner-validation scoring command; ticket 12 extends this to three 128³-only arms
with a Dice+CE candidate and separate sampling/loss comparisons. Both real pilot
outcomes are now complete: the user trained all three arms to epoch 50 and
evaluated 23 inner-validation examinations. Tickets 08 and 12 are completed
experiments, but neither candidate passed its predefined screen. Dice and FP
volume improved while small-lesion recall declined. See
[the results review](../../SAMPLING_PILOT_RESULTS.md). No new five-fold recipe
has been selected; the fourth loss-only arm remains a proposal.
Ticket 05 benchmark tooling is implemented, with GPU measurements still pending.
The missed-lesion/regression/SUVmax review is recorded in the follow-up report.
Do not treat sampler frequencies or synthetic tests as trained-model improvement.
