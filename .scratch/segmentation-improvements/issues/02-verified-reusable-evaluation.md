# 02: Make segmentation evaluation reusable and verified

**What to build:** Evaluate explicitly supplied native-grid predictions with one reproducible metric implementation usable by every experiment.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Expose explicit input/output configuration and reject mismatched case identities, invalid masks or incompatible image geometry.
- [ ] Report per-case Dice, lesion detection, unmatched-component FPV/FNV, TMTV and SUVmax, with aggregate results and size-stratified lesion recall.
- [ ] Document 18-connectivity, matching/tie rules, empty-mask behavior, units, size strata and the distinction between component and voxelwise errors.
- [ ] Pass synthetic known-geometry, empty-mask, split/merge, matching-tie and SUVmax-tie tests; preserve recorded upstream agreement where applicable.
- [ ] Record evaluator provenance and preserve input masks, quantitative SUV and existing outputs.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
