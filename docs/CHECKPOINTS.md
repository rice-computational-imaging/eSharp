# Checkpoints and GT evaluation

The authors selected the following calibrated checkpoints. Source and weights
are separate; download locations and checksums are not yet published. The
MVSEC Marigold step has a matching historical metric row; none of these full
GT evaluations has been independently reproduced by this release.

| Backend | Dataset | Selected step | GT evaluation config |
| --- | --- | ---: | --- |
| Lotus-D | DSEC | 34,000 | `configs/eval_dsec_gt.yaml` |
| Lotus-D | MVSEC | 19,500 | `configs/eval_mvsec_gt.yaml` |
| Marigold | DSEC | 7,000 | `config/eval_dsec_gt.yaml` |
| Marigold | MVSEC | 8,000 | `config/eval_mvsec_gt.yaml` |

DSEC uses `data_split/dsec_lidar_aligned/gt_val.txt`; MVSEC uses
`data_split/mvsec_pesudo_nbin3/gt_test.txt`. These point at real GT, preserving
the event/frame order of the author's selected splits. Depth encodings, GT,
camera calibration, and base models must be prepared separately; see DATA.md.
Pseudo-label supervision during training does not imply evaluation uses pseudo
labels. The saved `snapshot_*_pseudo_eval.yaml` files do use pseudo targets and
must not be reported as GT metrics.

## Local model layout

After downloading the models and configuring paths as in the root README:

```text
models/
  lotus-depth-d-v1-0/                 # complete base model, including VAE and text encoder
  marigold-v1-0-pretrained/stable-diffusion-2/
  esharp/
    lotus_dsec/step-00034000/         # model.safetensors, model_1.safetensors
    lotus_mvsec/step-00019500/        # model.safetensors, model_1.safetensors
    marigold_dsec/iter_007000/        # unet/diffusion_pytorch_model.bin, calibration_head.ckpt
    marigold_mvsec/iter_008000/       # unet/diffusion_pytorch_model.bin, calibration_head.ckpt
```

UNet config files may also be needed with the Marigold model. Use trusted
checkpoints only: Marigold's legacy `.bin` and `.ckpt` files use `torch.load`.
Optimizer, scheduler and trainer states are unnecessary for these evaluations.

## Evaluate calibrated checkpoints against GT

First configure local paths. For Lotus-D, set each `stage1.checkpoint` in
`configs/eval_dsec_gt.yaml` and `configs/eval_mvsec_gt.yaml` to the **same**
checkpoint passed to `--resume`: both the UNet and head must be loaded.

```bash
cd Lotus-ESharp-Fast
python train_diffused_lotus_d_calibration.py --config configs/eval_dsec_gt.yaml \
  --resume /absolute/location/models/esharp/lotus_dsec/step-00034000 --validate-only
python train_diffused_lotus_d_calibration.py --config configs/eval_mvsec_gt.yaml \
  --resume /absolute/location/models/esharp/lotus_mvsec/step-00019500 --validate-only
```

```bash
cd Marigold-ESharp
python train.py --config config/eval_dsec_gt.yaml \
  --init_from_ckpt /absolute/location/models/esharp/marigold_dsec/iter_007000 \
  --base_data_dir /absolute/location/data --base_ckpt_dir /absolute/location/models \
  --output_dir /absolute/location/runs/marigold-dsec-eval --eval_only --no_wandb --no_code_snapshot
python train.py --config config/eval_mvsec_gt.yaml \
  --init_from_ckpt /absolute/location/models/esharp/marigold_mvsec/iter_008000 \
  --base_data_dir /absolute/location/data --base_ckpt_dir /absolute/location/models \
  --output_dir /absolute/location/runs/marigold-mvsec-eval --eval_only --no_wandb --no_code_snapshot
```

Use distinct output directories. Both backends construct a training loader even
for evaluation; the configured training split must exist. Eval produces metrics
and configured visualizations. LPIPS is excluded from the release configs and
is not needed for their evaluation. Full protocol details, model licenses, data
availability and paper results still require confirmation.

## Evaluate Stage I without a calibration head

Stage-I checkpoints are valid to evaluate. They have no calibration head: use
`configs/eval_dsec_stage1_gt.yaml` or `configs/eval_mvsec_stage1_gt.yaml` with
Lotus's **`train_diffused_lotus_d.py`** and `--resume <stage1-checkpoint>
--validate-only`. A Stage-I Lotus checkpoint needs `model.safetensors`.

For Marigold use `config/eval_dsec_stage1_gt.yaml` or
`config/eval_mvsec_stage1_gt.yaml` with the same Marigold command above and
`--init_from_ckpt <stage1-checkpoint>`. These configs disable calibration. A
calibration-enabled eval config instead requires a trained head, and fails if
neither a head file nor a head in trainer state is present. Initializing new
Stage-II **training** from a Stage-I checkpoint still permits a new head.

## Publishing weights later

GitHub limits regular Git files and release assets; large model files may need
an external model host or supported sharding. Publish exact files, licenses,
URLs and SHA-256 checksums after final evaluation. See GitHub's
[large-file](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
and [release asset](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
documentation. Avoid publishing optimizer/trainer states without checking for
personal paths and other metadata.
