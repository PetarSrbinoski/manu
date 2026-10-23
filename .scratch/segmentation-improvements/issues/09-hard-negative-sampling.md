# 09: Evaluate hard-negative sampling

**What to build:** Mine false-positive regions from training patients and test whether sampling them improves rejection of difficult background.

**Blocked by:** 02: Make segmentation evaluation reusable and verified; 07: Audit training-patch coverage.

**Status:** ready-for-agent

- [ ] Record the mining model, training-patient provenance, candidate definitions and mixture with ordinary foreground/background samples.
- [ ] Verify that inner-validation and outer-test patients never enter the mined training pool; audit suspicious annotations.
- [ ] Exercise mining, sampling, training and prediction in a bounded comparison with matching reference settings.
- [ ] Report false-positive burden and retained small-lesion recall, including mining cost and representative errors.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
