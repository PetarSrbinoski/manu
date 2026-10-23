# 10: Evaluate multiple patches per examination

**What to build:** Expose training to several independently sampled and augmented patches per examination and quantify the benefit and cost.

**Blocked by:** 02: Make segmentation evaluation reusable and verified; 07: Audit training-patch coverage.

**Status:** ready-for-agent

- [ ] Make patch count configurable while retaining one patch per examination as the reference.
- [ ] Verify independent random transforms, deterministic-only preprocessing caching and compatible checkpoint/resume state.
- [ ] Run a bounded comparison reporting optimizer updates, sampled exposure and elapsed time rather than treating equal epochs as equal work.
- [ ] Evaluate small-lesion recall, false positives and whole-mask measures on the same inner-validation patients.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
