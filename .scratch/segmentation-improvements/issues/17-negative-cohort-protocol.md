# 17: Define a lesion-negative cohort experiment

**What to build:** Produce a reviewable protocol for extending the cohort with lesion-negative examinations, without initiating expansion.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Specify eligibility, data provenance, patient deduplication and handling of visits across positive and negative examinations.
- [ ] Define negative-case sampling, empty-mask behavior, split design and false-positive reporting on negative examinations.
- [ ] Distinguish lesion-negative examinations from healthy patients and identify expected data/compute needs.
- [ ] List implementation and data-access prerequisites for a separately authorized experiment; no downloads or training are part of this ticket.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
