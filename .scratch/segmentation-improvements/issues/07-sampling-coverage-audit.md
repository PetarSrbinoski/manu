# 07: Audit training-patch coverage

**What to build:** Measure which lesions and difficult background regions the current training sampler exposes to the model.

**Blocked by:** None (can start immediately).

**Status:** complete — bounded training-cohort audit and comparison reviewed

- [x] Use training patients only and retain the current 2:1 foreground/background policy as the reference.
- [x] Record coverage by reference component and predefined lesion-size groups, repeated draws and difficult high-uptake background exposure.
- [x] Use a fixed seed and bounded sample budget, preserving uncached random augmentation and original images.
- [x] Deliver a reproducible coverage report with examples and specific sampling hypotheses; document how difficult background was identified.

**Evidence (2026-10-25):** Six seeded fold-0 training examinations, 50 draws per
examination/policy at 128³, 300 draws per arm. The reference audit additionally
covered 64³/96³. The verified comparison includes 118 training-grid components:
53 <1 mL, 49 at 1–<10 mL and 16 ≥10 mL. All 53 small components appeared at least
once in both arms, but only 3/204 reference foreground draws targeted them versus
83/207 for uniform component sampling. Small-component mean visibility was
52.75% versus 54.04%; this does not establish a trained-model recall gain.

The actual MONAI cached/uncached paths are regression-tested. Early balanced
diagnostics were marked superseded after an uncached candidate-list regeneration
bug was discovered and fixed; use the verified comparison described in the
follow-up review. High-SUV background remains a proxy requiring visual review.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
