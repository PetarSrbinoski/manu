# 18: Select and freeze the next scientific recipe

**What to build:** Make a documented research decision and freeze a reproducible recipe for the next five-fold experiment.

**Blocked by:** 03: Certify the completed B baseline; 04: Review learning curves and inner-validation errors; additionally the explicitly selected pilot tickets from 06, 08–12, 14–16; selection must be recorded before work starts.

**Status:** ready-for-agent

- [ ] Before starting selection, explicitly record which pilot tickets from 6, 8–12, 14–16 belong to this round; those tickets are additional blockers until complete.
- [ ] Use inner-validation evidence and predefined clinical/research trade-offs; record rejected alternatives and unresolved limitations.
- [ ] Freeze scientific settings, patient assignments, seeds, checkpoint selection, stopping policy, budgets and evaluation criteria.
- [ ] Optional pilots outside the selected round do not block completion; mark follow-ups separately and acknowledge prior outer-result exposure.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
