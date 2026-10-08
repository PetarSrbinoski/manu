from pathlib import Path
import json
import os
import shutil
import random
from collections import defaultdict

# ovoa samo gi deli na folds za treniranje

# ============================================================
# SETTINGS
# ============================================================

PROJECT = Path(__file__).resolve().parent

# Tvojot download
SOURCE = PROJECT / "data" / "autopet-v2-lymphoma"

# Dataset sto ke go koristi training pipeline
OUTPUT = PROJECT / "data" / "lymphoma-baseline"

IMAGES_TR = OUTPUT / "imagesTr"
LABELS_TR = OUTPUT / "labelsTr"

N_FOLDS = 5
SEED = 42


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

IMAGES_TR.mkdir(parents=True, exist_ok=True)
LABELS_TR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER:
# hard-link instead of copying another 17 GB if possible
# ============================================================

def link_or_copy(source: Path, destination: Path):

    if destination.exists():
        return

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


# ============================================================
# FIND ALL PATIENTS + EXAMINATIONS
# ============================================================

cases = []

patients = sorted([
    p for p in SOURCE.iterdir()
    if p.is_dir()
])

for patient in patients:

    exams = sorted([
        e for e in patient.iterdir()
        if e.is_dir()
    ])

    for exam in exams:

        ct = exam / "CTres.nii.gz"
        suv = exam / "SUV.nii.gz"
        seg = exam / "SEG.nii.gz"

        if not ct.exists():
            raise FileNotFoundError(ct)

        if not suv.exists():
            raise FileNotFoundError(suv)

        if not seg.exists():
            raise FileNotFoundError(seg)

        # Unique examination/case ID.
        # Includes patient AND examination so that the patient
        # with two examinations gets two different case IDs.
        case_id = f"{patient.name}_{exam.name}"

        cases.append({
            "patient": patient.name,
            "exam": exam.name,
            "case_id": case_id,
            "ct": ct,
            "suv": suv,
            "seg": seg,
        })


print()
print("Found:")
print("Patients:    ", len(patients))
print("Examinations:", len(cases))


# ============================================================
# CHECK EXPECTED COUNTS
# ============================================================

if len(patients) != 144:
    print(
        f"WARNING: Expected 144 patients, "
        f"but found {len(patients)}"
    )

if len(cases) != 145:
    print(
        f"WARNING: Expected 145 examinations, "
        f"but found {len(cases)}"
    )


# ============================================================
# CREATE PATIENT-LEVEL 5 FOLDS
#
# IMPORTANT:
# We shuffle PATIENTS, not examinations.
#
# Therefore if one patient has two scans,
# both scans automatically get the same fold.
# ============================================================

patient_ids = [p.name for p in patients]

rng = random.Random(SEED)
rng.shuffle(patient_ids)


patient_to_fold = {}

for index, patient_id in enumerate(patient_ids):

    fold = index % N_FOLDS

    patient_to_fold[patient_id] = fold


# ============================================================
# ASSIGN EACH EXAMINATION ITS PATIENT'S FOLD
# ============================================================

for case in cases:

    case["fold"] = patient_to_fold[case["patient"]]


# ============================================================
# PREPARE FILES
#
# _0000 = CT
# _0001 = PET/SUV
# label = SEG
# ============================================================

print()
print("Preparing training dataset...")
print()

for number, case in enumerate(cases, start=1):

    case_id = case["case_id"]

    ct_destination = (
        IMAGES_TR / f"{case_id}_0000.nii.gz"
    )

    suv_destination = (
        IMAGES_TR / f"{case_id}_0001.nii.gz"
    )

    seg_destination = (
        LABELS_TR / f"{case_id}.nii.gz"
    )

    link_or_copy(
        case["ct"],
        ct_destination
    )

    link_or_copy(
        case["suv"],
        suv_destination
    )

    link_or_copy(
        case["seg"],
        seg_destination
    )

    print(
        f"[{number:3}/{len(cases)}] "
        f"Fold {case['fold']} | "
        f"{case['patient']} | "
        f"{case['exam']}"
    )


# ============================================================
# SAVE FOLD INFORMATION
# ============================================================

folds = []

for fold in range(N_FOLDS):

    validation = [
        case["case_id"]
        for case in cases
        if case["fold"] == fold
    ]

    training = [
        case["case_id"]
        for case in cases
        if case["fold"] != fold
    ]

    folds.append({
        "fold": fold,
        "training": training,
        "validation": validation,
    })


with open(
    OUTPUT / "splits_patient_level.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        folds,
        f,
        indent=4
    )


# ============================================================
# SAVE CASE INFORMATION
# ============================================================

case_info = []

for case in cases:

    case_info.append({
        "case_id": case["case_id"],
        "patient": case["patient"],
        "exam": case["exam"],
        "fold": case["fold"],
    })


with open(
    OUTPUT / "cases.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        case_info,
        f,
        indent=4
    )


# ============================================================
# VALIDATE PATIENT LEAKAGE
# ============================================================

patient_folds = defaultdict(set)

for case in cases:

    patient_folds[case["patient"]].add(
        case["fold"]
    )


leaks = {
    patient: fold_set
    for patient, fold_set in patient_folds.items()
    if len(fold_set) > 1
}


# ============================================================
# PRINT FOLD SUMMARY
# ============================================================

print()
print("=" * 70)
print("5-FOLD PATIENT-LEVEL SPLIT")
print("=" * 70)

for fold in range(N_FOLDS):

    fold_cases = [
        case
        for case in cases
        if case["fold"] == fold
    ]

    fold_patients = {
        case["patient"]
        for case in fold_cases
    }

    print(
        f"Fold {fold}: "
        f"{len(fold_patients)} patients, "
        f"{len(fold_cases)} examinations"
    )


# ============================================================
# CHECK PATIENT WITH MULTIPLE EXAMINATIONS
# ============================================================

patient_exam_count = defaultdict(int)

for case in cases:
    patient_exam_count[case["patient"]] += 1


multiple_exam_patients = {
    patient: count
    for patient, count in patient_exam_count.items()
    if count > 1
}


print()
print("Patients with multiple examinations:")

for patient, count in multiple_exam_patients.items():

    print(
        f"  {patient}: "
        f"{count} examinations -> "
        f"Fold {patient_to_fold[patient]}"
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 70)

if leaks:

    print("ERROR: PATIENT LEAKAGE DETECTED!")

    for patient, fold_set in leaks.items():
        print(
            patient,
            "appears in folds:",
            sorted(fold_set)
        )

else:

    print("NO PATIENT LEAKAGE")


expected_images = len(cases) * 2
expected_labels = len(cases)

actual_images = len(
    list(IMAGES_TR.glob("*.nii.gz"))
)

actual_labels = len(
    list(LABELS_TR.glob("*.nii.gz"))
)


print()
print("Files:")
print(
    f"imagesTr: {actual_images} "
    f"(expected {expected_images})"
)

print(
    f"labelsTr: {actual_labels} "
    f"(expected {expected_labels})"
)


if (
    actual_images == expected_images
    and actual_labels == expected_labels
    and not leaks
):

    print()
    print("DATASET READY FOR 5-FOLD TRAINING")

else:

    print()
    print("DATASET CHECK FAILED")


print("=" * 70)