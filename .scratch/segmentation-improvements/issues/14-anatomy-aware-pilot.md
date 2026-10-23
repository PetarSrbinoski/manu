# 14: Evaluate one anatomy-aware model variant

**What to build:** Test either organ-mask input channels or auxiliary organ supervision through a complete bounded train-to-evaluation path.

**Blocked by:** 02: Make segmentation evaluation reusable and verified; 13: Prepare and validate anatomical organ information.

**Status:** ready-for-agent

- [ ] Choose one variant and define its reference, hypothesis and additional inputs or loss weighting before launch.
- [ ] Verify train/inference compatibility, checkpoint identity and geometry; provide organ inputs at inference if the chosen variant requires them.
- [ ] Hold unrelated settings fixed and evaluate on the same separate inner-validation patients.
- [ ] Report physiological-region false positives, retained lesion recall, Dice and computational cost with overlays.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
