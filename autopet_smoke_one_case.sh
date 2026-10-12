#!/usr/bin/env bash
# Smoke test: official pretrained autoPET III model on ONE lymphoma examination.
# Official settings: 5-fold ensemble, mirroring TTA, step size 0.5.
#
# Run inside WSL from the project root:
#     bash autopet_smoke_one_case.sh                # default case below
#     bash autopet_smoke_one_case.sh "<case_id>"    # a case_id from cases.json
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HOME/venvs/autopet3"
DATA="$HOME/data/lymphoma-baseline"
MODEL="$PROJECT/checkpoints/autopet3/Dataset222_AutoPETIII_2024/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_bs3"
WORK="$HOME/autopet3_runs/smoke"

# One of the shortest examinations (284 slices), so the test is as quick as it can be.
CASE="${1:-PETCT_04ab5c61c9_11-22-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42241}"

for channel in 0000 0001; do
    if [ ! -f "$DATA/imagesTr/${CASE}_${channel}.nii.gz" ]; then
        echo "Missing: $DATA/imagesTr/${CASE}_${channel}.nii.gz" >&2
        exit 1
    fi
done

# nnU-Net predicts every case in the input folder, so the folder holds links to this case only.
rm -rf "$WORK/input"
mkdir -p "$WORK/input" "$WORK/output"
ln -s "$DATA/imagesTr/${CASE}_0000.nii.gz" "$WORK/input/"   # CT
ln -s "$DATA/imagesTr/${CASE}_0001.nii.gz" "$WORK/input/"   # PET (SUV)

# torch >= 2.6 refuses nnU-Net checkpoints in torch.load unless this is set. The checkpoint
# was verified against the Zenodo MD5, and the official repo stays unmodified this way.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "Case:   $CASE"
echo "Output: $WORK/output"
time "$VENV/bin/nnUNetv2_predict_from_modelfolder" \
    -i "$WORK/input" -o "$WORK/output" -m "$MODEL" \
    -npp 1 -nps 1 2>&1 | tee "$WORK/smoke.log"

ls -la "$WORK/output"
