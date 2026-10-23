# 21: Design an independent pretrained-model comparison

**What to build:** Deliver a feasible protocol for comparing models without the training-exposure problem of the current autoPET subset.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Assess an untouched external cohort and a comparator retrained within patient splits as alternative study designs.
- [ ] Specify training-data overlap evidence, patient eligibility, access requirements, preprocessing and common metric definitions.
- [ ] Predefine endpoints, uncertainty reporting and cohort/compute feasibility; identify missing provenance explicitly.
- [ ] Keep the current overlapping checkpoint results exploratory; collection, downloads and training require separate authorization.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
