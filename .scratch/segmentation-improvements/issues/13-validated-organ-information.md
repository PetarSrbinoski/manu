# 13: Prepare and validate anatomical organ information

**What to build:** Produce a small, aligned organ-information package suitable for anatomy-aware experiments on training and inner-validation cases.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Agree the organ-model source and available inputs; record weights/code provenance and obtain authorization for required downloads or inference.
- [ ] Extract a bounded sample and validate labels, native geometry and CT/PET alignment without modifying originals.
- [ ] Review representative organ masks and record failures, omissions, runtime and memory.
- [ ] Document reproducible preprocessing and output conventions usable at both training and inference.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
