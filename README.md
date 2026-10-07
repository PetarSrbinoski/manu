# MANU — lymphoma PET/CT project

Each examination contains:

| File | Purpose |
|---|---|
| `CTres.nii.gz` | CT anatomy, resampled to the PET image grid |
| `SUV.nii.gz` | PET tracer uptake, expressed as SUV |
| `SEG.nii.gz` | Reference mask marking the lesions |

## How it works

The script downloads selected files from inside the [official FDAT version-2 ZIP](https://doi.org/10.57754/FDAT.8f14a-pf846) without downloading the whole archive:

1. **Find lymphoma examinations.** Read the [official clinical metadata](https://www.cancerimagingarchive.net/wp-content/uploads/Clinical-Metadata-FDG-PET_CT-Lesions.csv), select rows labelled `LYMPHOMA`, and collect their patient and examination IDs. Multiple series rows belonging to the same examination are grouped together.
2. **Read the ZIP's index.** This small table of contents lists each file's name, size and location inside the archive.
3. **Select the required files.** For each lymphoma examination, select only `CTres.nii.gz`, `SUV.nii.gz` and `SEG.nii.gz`.
4. **Request only those bytes.** HTTP *range requests* tell the server: “send bytes X through Y.” Each request retrieves the part of the archive containing a selected file.
5. **Save the file.** Remove the outer ZIP compression while keeping the `.nii.gz` file intact. Check the downloaded size and checksum, then move the temporary `.part` file to its final filename.

This retrieved **16.62 GB of lymphoma files instead of the entire 330 GB archive**. Negative examinations and other cancer diagnoses were excluded. Version 2 uses the updated SUV calculation.
