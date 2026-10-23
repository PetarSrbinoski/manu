# 16: Establish an nnU-Net comparator pilot

**What to build:** Run a bounded nnU-Net comparison with the same patient assignments and independent inner-validation checkpoint selection.

**Blocked by:** 02: Make segmentation evaluation reusable and verified.

**Status:** ready-for-agent

- [ ] Record code/environment, CT/PET channel order, preprocessing, architecture and compute budget.
- [ ] Audit preprocessing for alignment, quantitative input handling and retained reference lesions; do not adopt a crop that excludes lesions.
- [ ] Verify patient routing and native-grid prediction export through a bounded train–select–predict–evaluate run.
- [ ] Report shared inner-validation metrics and relevant training differences without claiming an independent advantage from a one-fold pilot.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
