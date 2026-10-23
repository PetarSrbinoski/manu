# 15: Evaluate conservative postprocessing

**What to build:** Evaluate one conservative prediction-cleanup rule with an explicit accounting of true lesions it removes.

**Blocked by:** 02: Make segmentation evaluation reusable and verified; additionally 13 for an organ-aware rule (not required for component-only processing).

**Status:** ready-for-agent

- [ ] Choose an organ-aware or component-based rule and select its parameters using inner validation only.
- [ ] For an organ-aware rule, ticket 13 must complete before starting; component-only rules do not require it.
- [ ] Keep raw predictions immutable and verify deterministic geometry-preserving processing and edge-case behavior.
- [ ] Report before/after metrics and every removed reference lesion in the reviewed sample; do not blindly remove all organ-region or small components.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
