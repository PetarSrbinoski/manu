# 03: Certify the completed B baseline

**What to build:** Produce an evidence-backed baseline verification report from the recovered experiment and shared evaluator.

**Blocked by:** 01: Recover and inventory the completed B experiment; 02: Make segmentation evaluation reusable and verified.

**Status:** ready-for-agent

- [ ] Verify patient separation, repeated-visit grouping and checkpoint selection using inner validation only.
- [ ] Verify a consistent scientific recipe across all folds and trace each selected checkpoint to its history and provenance.
- [ ] Verify exactly one native-grid out-of-fold prediction for each of 145 examinations, and recompute per-case and aggregate metrics.
- [ ] Reconcile differences from published tables and identify any unresolved requirement; do not claim certification while required evidence is missing.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
