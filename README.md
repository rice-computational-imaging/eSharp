# eSharp: Diffusion-based Sharp Monocular Metric Event-to-Depth Estimation

eSharp adapts pretrained diffusion models for event-only monocular metric depth
estimation. It uses LiDAR-aligned dense pseudo-labels from a co-registered frame
camera for cross-modal distillation. This release contains the event-depth
training implementations used by eSharp:
`Marigold-ESharp` and `Lotus-ESharp-Fast`. They include DSEC and MVSEC data
configurations for pseudo-depth training and GT evaluation. Model weights, prepared datasets,
checkpoints, and experiment logs are intentionally not included.

Paper link: to be added when available.

## Pipeline

<img width="924" height="482" alt="Screenshot 2026-11-01 at 10 43 31 PM" src="https://github.com/user-attachments/assets/77ee0c34-ea59-49f4-9c13-81baa6539750" />

## Qualitative results

this might needs to be changed due to retrained marigold

DSEC

<img width="924" height="457" alt="Screenshot 2026-11-01 at 10 46 07 PM" src="https://github.com/user-attachments/assets/d86db78a-b831-490d-9507-f7a42bed80de" />

MVSEC

<img width="924" height="457" alt="Screenshot 2026-11-01 at 10 46 18 PM" src="https://github.com/user-attachments/assets/def88560-dae7-4d27-b027-b80d1c7696d8" />

Zero Shot Result

<img width="469" height="430" alt="Screenshot 2026-11-01 at 10 46 33 PM" src="https://github.com/user-attachments/assets/92debb02-0b67-4f9d-8d7c-d8a74f54905e" />


## Quantitative results
<img width="924" height="526" alt="Screenshot 2026-11-01 at 10 44 09 PM" src="https://github.com/user-attachments/assets/1f6e35a2-762c-47ab-9bc2-fb99a523c890" /><img width="924" height="253" alt="Screenshot 2026-11-01 at 10 44 58 PM" src="https://github.com/user-attachments/assets/f174c5a9-37ab-41d3-899e-01412f6fc7f6" />


## Layout

| Location | Purpose |
| --- | --- |
| `Marigold-ESharp/` | Event-conditioned Marigold training, validation, and data loaders. |
| `Lotus-ESharp-Fast/` | Event-conditioned Lotus-D training and calibration. |
| `Marigold-ESharp/config/dataset/dsec_pseudo.yaml` | Shared DA2 LiDAR-aligned DSEC split. |
| `Marigold-ESharp/config/esharp/snapshot_marigold_dsec_stage2.yaml` | Frozen Marigold DSEC release configuration. |
| `Lotus-ESharp-Fast/configs/esharp/` | Historical Lotus-D checkpoint configurations. |
| `docs/CHECKPOINTS.md` | Required checkpoint structure and evaluation commands. |

MVSEC YAMLs are retained under both backend configuration directories. Historical
non-DA2 DSEC label configurations are deliberately not part of the active release.

The implementation uses the existing `mvsec` dataset loader name for these DSEC
event/pseudo-depth records. This is a loader name inherited from development, not
a claim that DSEC data are MVSEC data.

## Main configurations

| Backend | Dataset | Stage I | Stage II | GT evaluation |
| --- | --- | --- | --- | --- |
| Lotus-D | DSEC | `configs/train_dsec_stage1.yaml` | `configs/train_dsec_stage2.yaml` | `configs/eval_dsec_gt.yaml` |
| Lotus-D | MVSEC | `configs/train_mvsec_stage1.yaml` | `configs/train_mvsec_stage2.yaml` | `configs/eval_mvsec_gt.yaml` |
| Marigold | DSEC | `config/train_dsec_stage1.yaml` | `config/train_dsec_stage2.yaml` | `config/eval_dsec_gt.yaml` |
| Marigold | MVSEC | `config/train_mvsec_stage1.yaml` | `config/train_mvsec_stage2.yaml` | `config/eval_mvsec_gt.yaml` |

Paths in this table are relative to the corresponding backend. The main Stage-I
and Stage-II training configs select pseudo-depth targets; the separate GT
evaluation configs select real GT. Training does not combine pseudo-label
backbone supervision with a separate GT calibration target. The `esharp/` configs
are historical checkpoint snapshots; their saved validation targets can be pseudo
labels. The separate `eval_*_gt.yaml` configs select real GT for validation.
`snapshot_*` files record historical checkpoint settings; they are not
necessarily the paper's final protocol. Dataset YAMLs under `config/dataset/`
define the loader and split; they are inherited by the Marigold configs.

Both current calibration implementations average masked L1 error within each
image and then across images. This change to Lotus affects future training;
existing weights were not retrained or re-evaluated as part of this update.

To evaluate a Stage-I checkpoint that has no calibration head, use the
`eval_dsec_stage1_gt.yaml` or `eval_mvsec_stage1_gt.yaml` config in the relevant
backend. For Lotus-D use `train_diffused_lotus_d.py`; for Marigold use `train.py`.
Calibrated evaluation uses the regular `eval_*_gt.yaml` configs and requires the
saved calibration head. See [checkpoint commands](docs/CHECKPOINTS.md).

## Environment

Use Linux, Python 3.10, and a CUDA-enabled PyTorch installation. The two code
directories must remain together: Lotus reads the DSEC split through the included
Marigold loader.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-event.txt
python -m pip check
python -c "import torch, torchvision, xformers, tabulate, tifffile; print('CUDA available:', torch.cuda.is_available())"
python tools/check_source.py
python -m unittest discover -s tests -p 'test_evaluation_safety.py'
```

Marigold enables xFormers attention unconditionally. The version in the
requirements matches the preparation server's PyTorch pair; use an appropriate
CUDA driver and validate a fresh environment before relying on these pins. Some
auxiliary figure scripts need extra projects and packages outside this list.

Prepare data, base models, and run outputs outside the repository. Placeholder
paths in the YAML files can be configured with:

```bash
python tools/configure_paths.py \
  --data-root /absolute/location/data \
  --model-root /absolute/location/models \
  --run-root /absolute/location/runs \
  --dsec-pseudo-root /absolute/location/dsec_da2_lidar_pseudo_labels \
  --dry-run
# Repeat without --dry-run to configure this local working copy.
```

The data root must contain the paths referenced by
`Marigold-ESharp/data_split/dsec_lidar_aligned/`. These split files pair DSEC
`nbin_3_encoding` events with the DA2 LiDAR-aligned dense pseudo-depth maps.
See [pseudo-label generation](docs/PSEUDO_LABELS.md) for the DSEC generator,
MVSEC quantile alignment, input requirements and validation limitations.

The pseudo-label root must contain one subdirectory per DSEC sequence, with the
`frame..._pred_depth.npy` files named in the split lists. `--data-root` alone
does not configure this separate root. If DSEC pseudo-labels are not needed,
the extra argument can be omitted and the unresolved-path warning is expected.
Use paths without spaces, quotes, backslashes, `$`, backticks or `#`. Keep the
placeholder source copy for publication; the tool modifies files in place.

See [data setup](docs/DATA.md) for the split format, GT loaders, and missing
preprocessing requirements, and [release status](docs/RELEASE_STATUS.md) for
the remaining checks before publication.

## Training

Run the commands from the relevant backend directory after configuring local paths.
Always use a new output directory for a new experiment.

```bash
cd Lotus-ESharp-Fast
CUDA_VISIBLE_DEVICES=0 python train_diffused_lotus_d.py \
  --config configs/train_dsec_stage1.yaml

python train_diffused_lotus_d_calibration.py \
  --config configs/train_dsec_stage2.yaml
```

```bash
cd Marigold-ESharp
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config config/train_dsec_stage1.yaml \
  --base_data_dir /absolute/location/data \
  --base_ckpt_dir /absolute/location/models \
  --output_dir /absolute/location/runs/marigold-dsec --no_wandb
```

For Stage II, set the Lotus `stage1.checkpoint` to the completed matching Stage-I
run. For Marigold, use `--init_from_ckpt` to start a new calibration run from a
saved Stage-I model. `--resume`/`--resume_run` are only for continuing the exact
same run with its existing optimizer state.

## Attribution

This release builds on Marigold and Lotus. Their licenses and upstream attribution
files are retained in each backend directory.

## Citation

The final paper citation has not yet been supplied. Preserve the upstream
Marigold and Lotus attribution when adding the ESharp citation.
