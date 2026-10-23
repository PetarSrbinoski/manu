# 20: Measure training variability with additional seeds

**What to build:** Quantify how much the frozen recipe changes with independent training randomness.

**Blocked by:** 19: Complete and verify the frozen five-fold experiment.

**Status:** ready-for-agent

- [ ] Predefine the number of additional seeds and resource budget; keep patient assignments and the scientific recipe fixed.
- [ ] Run separately authorized seed experiments in isolated outputs and verify their complete out-of-fold predictions.
- [ ] Report across-seed variability separately from patient-sampling uncertainty and compare failure patterns.
- [ ] Do not select the best seed on outer scores or ensemble models that trained on an evaluated patient.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
