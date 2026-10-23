# 05: Benchmark feasible training patch sizes

**What to build:** Produce bounded memory and throughput measurements that identify feasible contextual patch sizes on the available hardware.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Benchmark 64³, 96³ and 128³ with fixed 2 mm spacing and architecture; record hardware and all resolved settings.
- [ ] Measure training and sliding-window inference memory/time separately, recording failures without silently changing the recipe.
- [ ] Use isolated diagnostics; test 160³/192³ only on suitable available hardware.
- [ ] Deliver a feasible-size recommendation and exact bounded reproduction settings; do not claim segmentation improvement from throughput measurements.

**Experiment constraints:** Preserve original images and saved experiments. Use separate outputs and record provenance. Keep training, inner validation and outer test patients separate, including repeat visits. Use inner validation for tuning, preserve native geometry/full field of view and quantitative SUV, and do not silently change the scientific recipe to fit hardware. Ticket approval authorizes planning, not full training, dataset/checkpoint downloads or large installations. Before any such launch, verify existing session authorization; if absent, prepare a concrete run specification for approval. Implementation or smoke-test success alone does not complete a ticket whose deliverable includes experiment results.
