# Data and evaluation setup

The repository contains split lists, not event data, pseudo-labels, GT maps or
weights. See [pseudo-label generation](PSEUDO_LABELS.md) for the DSEC generator
and MVSEC teacher-inference and quantile-alignment steps. Raw-data preparation
and event binning are not yet included as a complete reproducible recipe.
Public download locations and checksums must be provided before release.

## Split format

The retained loaders expect one whitespace-separated event/depth pair per line.
Event paths may be relative to the configured data root; absolute paths are used
as written. The DSEC pseudo-label splits refer to a separate pseudo-depth root;
configure it with `--dsec-pseudo-root` as described in the root README.

The package retains one split directory per dataset: `dsec_lidar_aligned/`
and `mvsec_pesudo_nbin3/`. DSEC's `gt_train.txt`, `gt_val.txt`, and
`gt_val_first.txt` preserve the original GT evaluation frame order and refer
directly to disparity PNGs; the other lists point at aligned pseudo-depth.
MVSEC `gt_train.txt`, `gt_test.txt`, and `gt_test_first.txt` similarly refer
directly to GT depth arrays. Keep the supplied lists when
reproducing results: a generator alone cannot guarantee the same examples or
their order. A complete raw-data-to-split generator is not yet available.

The historical loader named `mvsec` reads the `.npy` pseudo-depth path in the
second column, for DSEC as well as MVSEC. `mvsec_gt` reads an explicit GT
`depth_*.npy` path; `dsec_event_disparity` reads an explicit disparity PNG.
Both also accept legacy pseudo-label paths and derive the GT filename from the
sequence and frame index. A GT config can affect its training split too.
Direct GT rows do not add a `pseudo_depth_linear` visualization sidecar;
legacy rows can still provide one when the pseudo-depth file is available.

## Generate a split from prepared files

`tools/generate_splits.py` pairs prepared event files with existing pseudo-labels
or GT for explicitly chosen sequences. It does not create the underlying data,
choose train/test sequences, or infer timestamps and camera alignment. Select
disjoint sequences and verify the index mapping yourself; incorrect mapping can
produce plausible-looking but misaligned pairs.

For example, given prepared DSEC TIFFs, pseudo labels, and a verified mapping
where each event frame index maps to twice that index in the pseudo labels:

```bash
python tools/generate_splits.py pair --dataset dsec --target pseudo \
  --events-root /absolute/data/event3d/DSEC/train/nbin_3_encoding \
  --pseudo-root /absolute/pseudo-depth --sequence zurich_city_05_a \
  --index-multiplier 2 --output /absolute/my_splits/val.txt
```

For GT use `--target gt --data-root /absolute/data`; repeat `--sequence` to
include more sequences. MVSEC naming and offsets must match the prepared
files; use the correct dataset, index multiplier and offset. The output directory
must already exist. By default missing targets fail rather than silently changing
the split; `--skip-missing` explicitly opts into dropping them. The generated
list is not automatically the paper split. Point your YAML `filenames` to it.

For existing lists, `convert-legacy-gt --dataset dsec|mvsec --input OLD --output NEW`
preserves order while converting legacy pseudo-style names to direct GT paths.

## Ground truth

After configuring `/path/to/data`, the default DSEC GT loader reads disparity
maps under `event3d/DSEC/train/disparity_maps/<sequence>/disparity/event/`
and per-sequence camera calibration under `event3d/DSEC/train/calib/`.
The MVSEC GT loader reads indexed depth arrays under
`event3d/mvsec/train/<sequence>/depth/data/`. It requires exact matching
frame indices. Some MVSEC day-2 sequence names map to nested directories;
see `src/dataset/mvsec_gt_depth_dataset.py`.

The GT evaluation configs select GT targets for their validation loaders. They
do not establish the final paper evaluation protocol. Confirm split, crop,
depth limits, alignment, calibrated head, and metrics before comparing results.
The evaluation code also constructs a training loader during initialization.

## Representation and limitations

Event inputs can be encoded uint8 images or normalized floating-point voxel
TIFFs; `event_input_mode` selects the loader behavior. Verify that the encoding
used by a config matches the prepared data. Auxiliary scripts under `script/`
may require separate projects, paths, packages or model weights; they are not
part of the validated training environment.

Calibration currently uses the selected training dataset's depth and mask as
its target. Loading pseudo-labels does not additionally load real GT for the
calibration loss. The frozen configs are evidence of run settings, not proof
that numerical paper results have been reproduced.
