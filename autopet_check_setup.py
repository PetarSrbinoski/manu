"""Check that the official autoPET III code can load the downloaded checkpoint. Runs NO inference.

Run inside WSL:
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 ~/venvs/autopet3/bin/python autopet_check_setup.py

It uses the repo's own nnUNetPredictor.initialize_from_trained_model_folder, the same call that
nnUNetv2_predict_from_modelfolder makes, then loads the weights of every fold into the network.
"""
import argparse
from pathlib import Path

import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

PROJECT = Path(__file__).resolve().parent
MODEL_DIR = (PROJECT / 'checkpoints' / 'autopet3' / 'Dataset222_AutoPETIII_2024'
             / 'autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_bs3')
FOLDS = (0, 1, 2, 3, 4)


def main(model_dir):
    print(f'torch {torch.__version__}, cuda available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}, '
              f'{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB')

    predictor = nnUNetPredictor(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    predictor.initialize_from_trained_model_folder(str(model_dir), FOLDS, 'checkpoint_final.pth')

    config = predictor.configuration_manager
    print(f'trainer:              {predictor.trainer_name}')
    print(f'network class:        {type(predictor.network).__module__}.{type(predictor.network).__name__}')
    print(f'channel_names:        {predictor.dataset_json["channel_names"]}')
    print(f'normalization:        {config.normalization_schemes}')
    print(f'spacing (z, y, x):    {config.spacing}')
    print(f'patch size:           {config.patch_size}')
    print(f'segmentation heads:   {predictor.label_manager.num_segmentation_heads} {predictor.dataset_json["labels"]}')
    print(f'TTA mirroring axes:   {predictor.allowed_mirroring_axes}')
    print(f'folds loaded:         {len(predictor.list_of_parameters)}')

    # load_state_dict is strict: it fails if any key of the checkpoint does not fit the network.
    for fold, parameters in zip(FOLDS, predictor.list_of_parameters):
        predictor.network.load_state_dict(parameters)
        print(f'fold_{fold}: weights fit the network ({len(parameters)} tensors)')
    print('OK - checkpoint and code are compatible. No inference was run.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    main(parser.parse_args().model_dir)
