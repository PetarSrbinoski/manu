# 12: Evaluate Dice plus cross-entropy loss

**What to build:** Provide and evaluate a configurable combined loss against the existing background-inclusive Dice loss.

**Blocked by:** 02: Make segmentation evaluation reusable and verified.

**Status:** ready-for-agent

- [ ] Specify loss weighting and retain the existing loss as the default reference; preserve float32 reductions and numerical recovery.
- [ ] Verify finite loss/gradients and resume compatibility for the selected loss configuration.
- [ ] Run a bounded comparison with identical patients, patch, sampling and optimization settings apart from the loss.
- [ ] Report false-positive reduction alongside small-lesion recall, Dice and volume/SUV errors.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
