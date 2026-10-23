# 11: Evaluate a revised learning-rate or stopping policy

**What to build:** Test one explicitly justified schedule or stopping-policy change against the reference in a separate experiment.

**Blocked by:** 02: Make segmentation evaluation reusable and verified; 04: Review learning curves and inner-validation errors.

**Status:** ready-for-agent

- [ ] Use curve-review evidence to specify one changed factor, a bounded budget and acceptance criteria before launch.
- [ ] Start fresh outputs and preserve the original cosine schedule and checkpoints; reject incompatible resume attempts.
- [ ] Verify schedule boundaries or stopping behavior and record actual learning rates, selected epochs and stopping reasons.
- [ ] Compare inner-validation masks and shared metrics without selecting settings on outer test patients.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
