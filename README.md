# MANU — lymphoma PET/CT project

Each examination contains:

| File | Purpose |
|---|---|
| `CTres.nii.gz` | CT anatomy, resampled to the PET image grid |
| `SUV.nii.gz` | PET tracer uptake, expressed as SUV |
| `SEG.nii.gz` | Reference mask marking the lesions |

## How it works

From the [official FDAT version-2 ZIP](https://doi.org/10.57754/FDAT.8f14a-pf846):

1. Find lymphoma examinations in the [official metadata](https://www.cancerimagingarchive.net/wp-content/uploads/Clinical-Metadata-FDG-PET_CT-Lesions.csv).
2. Read the ZIP index to find each file's location and size.
3. Select only CTres, SUV and SEG for those examinations.
4. Request only those bytes.
5. Extract and save the `.nii.gz` files into patient/examination folders.

Downloaded: 145 examinations from 144 patients, totaling 16.62 GB. Negative examinations and other cancer diagnoses are excluded.
