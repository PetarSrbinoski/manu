# Lymphoma segmentation with unlimited resources

This is a theoretical research direction, not an approved implementation or
training plan. It does not authorize new training, data downloads or cohort
expansion. The proposals below require controlled evaluation; none is guaranteed
to improve segmentation.

## Main direction

Build a large, carefully labelled, multicentre dataset and train a model that
combines whole-body context with detailed local segmentation. Validate it on
genuinely independent hospitals.

Our completed 128³ pilot demonstrates why Dice alone is insufficient. The
balanced-sampling plus Dice/CE model improved mean Dice from 0.516 to 0.565,
but detected only 17 of 348 small reference lesions, compared with 37 for the
original model. These are results from 23 inner-validation examinations in one
fold, not independent test performance. See [the pilot results](architecture-cleanup-and-pilot-results-summary.md).

The objective must account for finding lesions as well as drawing their boundaries.

## Priorities

| Investment | Proposed work | Why it matters |
|---|---|---|
| More representative data | Collect thousands of lymphoma PET/CT examinations across hospitals, scanners, disease subtypes and treatment stages, including lesion-negative scans. | Our 145 positive examinations cannot establish performance across that range or false positives on negative scans. |
| Expert-reviewed labels | Have multiple specialists review small lesions, ambiguous uptake and model disagreements, with adjudication and uncertainty recorded. | Training and evaluation depend on consistent labels for tiny lesions and difficult findings. |
| Whole-body context plus local detail | Combine a global anatomical representation with high-resolution local segmentation. | The model needs both the location of uptake in the body and local detail for small lesions. |
| More useful training examples | Sample multiple regions per examination, mixing ordinary patches, small-lesion patches and difficult background examples. | One patch per examination per epoch offers limited exposure; balanced sampling alone did not solve small-lesion detection. |
| Controlled model comparisons | Compare a strong nnU-Net baseline, larger residual U-Nets and alternative architectures under matched evaluation. | Architecture should earn its complexity through measured improvements. |
| Independent evaluation | Reserve hospitals and later-acquired examinations for a final locked evaluation. | This tests whether improvements survive outside the development dataset. |

The broader public FDG dataset includes positive and lesion-negative examinations.
Expanding our cohort would require a new dataset definition, provenance checks,
empty-mask support and patient splits. Lesion-negative does not mean healthy.
Verify the actual downloaded dataset version before treating records as
interchangeable. [Original dataset publication](https://www.nature.com/articles/s41597-022-01718-3.pdf)

## Proposed model design

Test two cooperating parts:

- A whole-body context model representing anatomy and the distribution of PET uptake.
- A local segmentation model examining overlapping regions in greater detail,
  using PET/CT and the global context.

Preserve coverage of the entire examination, CT/PET alignment, quantitative PET
SUV values and native output geometry. A detector examining only proposed
suspicious regions can permanently miss lesions excluded by its first stage.
Organ information should provide context rather than automatically delete
predictions inside organs. Finer processing grids do not recover information
absent from the original acquisition.

This design is a research hypothesis. Benchmark it against nnU-Net, an
established framework for configuring preprocessing, architecture, training and
postprocessing. [nnU-Net paper](https://www.nature.com/articles/s41592-020-01008-z)

## Training and model selection

Current checkpoints are selected using inner-validation Dice. The pilot shows
that a higher Dice score can coexist with worse lesion detection.

Before a new experiment, define a model-selection rule accounting for:

- Small-lesion sensitivity at an acceptable false-positive burden.
- False-positive counts and volume, including on negative examinations.
- Dice, tumour-volume error and SUVmax error as complementary measures.
- Performance across lesion sizes, hospitals and scanners.

Agree the acceptable tradeoff with Igor and clinical collaborators. Tune and
select using inner validation; keep outer test patients and the external test
set separate. Changing the selection rule is a new scientific experiment, not
a retrospective justification for choosing a favourable checkpoint.

Use abundant compute for controlled experiments across several seeds and for
training long enough to assess convergence. Compare sampling, loss, context,
resolution and architecture changes systematically. When increasing patches per
examination, record optimizer updates and compute budgets as well as epochs.
Specify schedules before launch; changing the total epoch budget changes the
current cosine learning-rate schedule.

Evaluate ensembles of complementary models if validation supports them. Every
ensemble member used to evaluate a patient must have excluded that patient from
training and model selection. Averaging the five existing outer-fold models on
the current cohort would leak information. Keep repeated visits of each patient
together in all partitions.

## What unlimited compute cannot replace

The largest priorities are data quality and diversity, useful small-lesion
exposure, and global anatomical context. Larger patches and more epochs are
experiment options, but we do not have evidence that they alone resolve the
observed failure pattern.

More compute can explore alternatives. Independent data and reliable labels are
needed to determine whether they work. Comparison with pretrained autoPET models
must document training-data overlap; overlapping public-data scores are
exploratory rather than independent evidence of generalization.

GenAI remains a separate later research task whose role, required data,
comparator and measurable endpoint must be agreed before implementation.
