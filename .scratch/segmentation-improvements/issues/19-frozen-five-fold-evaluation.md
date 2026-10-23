# 19: Complete and verify the frozen five-fold experiment

**What to build:** Complete the authorized frozen experiment and deliver a verified patient-level out-of-fold result package.

**Blocked by:** 18: Select and freeze the next scientific recipe.

**Status:** ready-for-agent

- [ ] Use a new output location and identical scientific settings for all folds; preserve progress logs, numerical safeguards and resume compatibility.
- [ ] Verify five selected checkpoints and completion records, strict patient separation and exactly 145 native-grid out-of-fold predictions.
- [ ] Report per-case/aggregate Dice, lesion-size results, false-positive burden, TMTV/SUVmax errors and patient-aware uncertainty.
- [ ] Include learning curves, representative errors, provenance and a completeness audit; never declare completion from partial folds.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
