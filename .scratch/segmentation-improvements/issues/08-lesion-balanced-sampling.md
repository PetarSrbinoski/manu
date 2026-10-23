# 08: Evaluate lesion-balanced sampling

**What to build:** Train and evaluate a configurable sampler that selects reference lesions before choosing foreground patch centres.

**Blocked by:** 02: Make segmentation evaluation reusable and verified; 07: Audit training-patch coverage.

**Status:** ready-for-agent

- [ ] Preserve background sampling, paired spatial transforms and the current sampler as a selectable reference.
- [ ] Verify component selection and patch targeting on synthetic cases containing lesions of different sizes.
- [ ] Run an authorized bounded inner-validation comparison with recorded update/time budgets and unchanged non-sampling settings.
- [ ] Report small-lesion recall, false positives and whole-mask metrics; reject unsupported claims based solely on sampling frequencies.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
