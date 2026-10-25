# 07: Audit training-patch coverage

**What to build:** Measure which lesions and difficult background regions the current training sampler exposes to the model.

**Blocked by:** None (can start immediately).

**Status:** implemented — awaiting explicit cohort audit execution

- [ ] Use training patients only and retain the current 2:1 foreground/background policy as the reference.
- [ ] Record coverage by reference component and predefined lesion-size groups, repeated draws and difficult high-uptake background exposure.
- [ ] Use a fixed seed and bounded sample budget, preserving uncached random augmentation and original images.
- [ ] Deliver a reproducible coverage report with examples and specific sampling hypotheses; document how difficult background was identified.

**Implementation evidence (2026-10-25):** A seeded, training-only audit carries
reference component IDs through the actual crop/pad/affine pipeline and reports
component appearances by size, repeated centres and high-SUV background exposure.
Synthetic tests verify deterministic coverage, known geometry and unchanged
reference inputs. Cohort results and the resulting sampling hypotheses remain
pending. High SUV outside annotations is a proxy requiring review, not confirmed
physiological anatomy or an automatically accepted hard negative.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
