# DiffusED: Event-Conditioned Metric Depth with Marigold

This repository is a research fork of the official Marigold codebase for event-camera depth estimation. It adapts the Marigold latent diffusion training pipeline to event inputs and adds a lightweight metric calibration stage for DSEC and MVSEC experiments.

The original Marigold license is kept in this repository. This README documents the DiffusED-specific code paths, configs, and commands; please see the upstream Marigold repository for the original project README.

## What This Fork Adds

- Event-conditioned Marigold fine-tuning with an 8-channel UNet input stem.
- DSEC and MVSEC dataset configs and split files.
- Old `nbin3` event encoding support for event-depth experiments.
- Dense pseudo-depth training from RGB-to-depth teacher predictions.
- Metric pseudo-label alignment utilities.
- A Stage-II calibration head that predicts per-frame affine scale and shift.
- Evaluation metrics compatible with the Event-WassDiff / DepthAnyEvent-style clipped metric protocol.
- SVG/figure-generation scripts for paper assets and TUM-VIE zero-shot examples.

## Method Summary

DiffusED uses Marigold as the base latent diffusion depth model. The RGB conditioning pathway is replaced by an event representation. In our main event setup, events are temporally binned into a three-channel representation, and the Stable Diffusion / Marigold UNet input convolution is expanded from 4 channels to 8 channels so that the noisy depth latent and event latent can be concatenated.

Training is two-stage:

1. **Stage I: event-conditioned diffusion fine-tuning.** The UNet is fine-tuned with the diffusion denoising objective against pseudo-depth supervision. The VAE and text encoder remain frozen.
2. **Stage II: metric calibration.** A compact calibration head predicts a global scale and shift for each predicted depth map. In the joint calibration configs, the calibration head and UNet are optimized together, while the VAE remains frozen.

The calibration implementation is in:

- `src/util/calibration.py`
- `src/trainer/marigold_trainer.py`

The pipeline implementation is in:

- `marigold/marigold_pipeline.py`

## Environment

The code has been used with Python 3.10 and CUDA GPUs. On the project server we typically use:

```bash
cd /path/to/repository/Marigold
source /path/to/repository/Marigold/venv/marigold/bin/activate

export BASE_DATA_DIR=/path/to/data
export BASE_CKPT_DIR=/path/to/models
```

For a new machine, create an environment from the Marigold requirements:

```bash
python -m venv venv/marigold
source venv/marigold/bin/activate
pip install -r requirements.txt -r requirements+.txt -r requirements++.txt
```

## Data, training, and evaluation

This is a historical fork overview. The supported release configuration,
installation, data layout, training, and evaluation instructions are in the
[release README](../README.md) and [data guide](../docs/DATA.md). DSEC uses
`config/dataset/dsec_pseudo.yaml`, and MVSEC uses
`config/dataset/mvsec_pseudo.yaml`. Historical configs and split
paths mentioned in earlier versions of this overview were removed from the release.

## Figure and Zero-Shot Scripts

Paper asset scripts live in:

```bash
script/figure_assets/
```

Useful scripts:

- `script/figure_assets/generate_dsec_pipeline_svgs.py`
- `script/figure_assets/generate_dsec_test_pipeline_svgs.py`
- `script/figure_assets/generate_mvsec_pipeline_svgs.py`
- `script/zero_shot_tumvie.py`

These scripts were written for our paper asset workflow and may contain local default paths. Treat them as reproducibility scripts for the lab environment unless paths are parameterized for a public release.

## Checkpoints

Large checkpoints are intentionally not committed to git. For reproducibility, store them externally and document their paths or download links.

During development, checkpoints were commonly stored under:

```bash
/path/to/data/marigold/
/path/to/models/
```

## Release Checklist

Before making this repository public, check:

- Replace or document lab-specific absolute paths in `data_split/`, configs, and figure scripts.
- Remove unused experiment files from the upstream fork if they are not part of the release.
- Confirm no checkpoints, output folders, generated predictions, or private data are tracked.
- Add checkpoint download instructions or a model card if releasing weights.
- Add citation information for Marigold, Depth Anything V2, DSEC, MVSEC, TUM-VIE, and any compared method used in the paper.
- Run at least a smoke evaluation on one DSEC and one MVSEC split after a fresh clone.

## Attribution

This code is based on the official Marigold repository:

```text
Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation
CVPR 2024
```

Please keep the original Marigold attribution and license when releasing this fork.
