# 01: Recover and inventory the completed B experiment

**What to build:** Recover the colleague’s completed testTraining experiment as an isolated, traceable package, without replacing the unfinished local run.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Inventory all five folds, source identity, configuration, patient splits, histories, completion records, metrics, migration records, checkpoints and predictions.
- [ ] Record checksums and identify missing or unreadable artifacts; distinguish recovered evidence from reported claims.
- [ ] Preserve originals and existing local outputs; document how to locate the recovered package.
- [ ] If artifacts are unavailable, record the specific handoff required; do not certify recovery as complete.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
