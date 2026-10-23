# 04: Review learning curves and inner-validation errors

**What to build:** Explain likely learning limitations using recovered histories and representative inner-validation predictions.

**Blocked by:** 01: Recover and inventory the completed B experiment.

**Status:** ready-for-agent

- [ ] Plot training loss, inner-validation foreground Dice and learning rate with selected and stopping epochs marked.
- [ ] Review small lesions, extensive disease, poor cases and physiological uptake using aligned CT/PET/reference/prediction overlays.
- [ ] Distinguish observations from hypotheses about inadequate learning, validation noise and overfitting.
- [ ] Document candidate experiments without tuning against outer test patients; identify any inference work requiring launch authorization.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
