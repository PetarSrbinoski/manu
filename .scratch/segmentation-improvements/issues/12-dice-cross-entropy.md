# 12: Evaluate Dice plus cross-entropy loss

**What to build:** Provide and evaluate a configurable combined loss against the existing background-inclusive Dice loss.

**Blocked by:** 02: Make segmentation evaluation reusable and verified.

**Status:** implemented and CPU-verified — real three-arm pilot/evaluation pending user launch

- [x] Specify loss weighting and retain the existing loss as the default reference; preserve float32 reductions and numerical recovery.
- [x] Verify finite loss/gradients and resume compatibility for the selected loss configuration.
- [ ] Run a bounded comparison with identical patients, patch, sampling and optimization settings apart from the loss.
- [ ] Report false-positive reduction alongside small-lesion recall, Dice and volume/SUV errors.

The third 128³ fold-0 pilot arm uses lesion-balanced sampling plus 1 × Dice +
1 × cross-entropy. A/B isolates sampling; B/C isolates loss. All pause at epoch
50 of the unchanged 500-epoch schedule before outer inference. Loss type and
weights are scientific checkpoint metadata and cannot be changed on resume.
The frozen C/B screen requires at least 10% lower mean unmatched FP volume,
at most 0.02 lower small-component recall and at most 0.02 lower Dice. CPU tests
verify weighted arithmetic, finite gradients, AMP-to-float32 recovery and
epoch-boundary resume. These tests do not establish segmentation improvement.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
