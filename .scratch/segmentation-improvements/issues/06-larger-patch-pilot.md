# 06: Evaluate a larger-patch training pilot

**What to build:** Compare the existing 64³ recipe with one feasible larger patch on the same training and inner-validation patients.

**Blocked by:** 02: Make segmentation evaluation reusable and verified; 05: Benchmark feasible training patch sizes.

**Status:** ready-for-agent

- [ ] Specify the hypothesis, update budget, stopping rules and success criteria before launch; preserve outer test patients.
- [ ] Hold other scientific settings fixed and record differences in sampled voxel exposure, optimizer updates and elapsed time.
- [ ] Complete the bounded comparison and report Dice, false-positive burden, small-lesion recall and TMTV/SUVmax errors with overlays.
- [ ] Document whether added context helped and the limits of a one-fold pilot; retain both experiment artifacts.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
