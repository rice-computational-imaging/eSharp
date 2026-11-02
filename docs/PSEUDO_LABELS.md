# Pseudo-depth labels

These tools generate training targets from prepared data. They do not produce event
bins or prepare synchronized, registered inputs from raw recordings. Use ground
truth from training sequences only to generate training targets; using evaluation
GT would leak information into training. Keep the supplied evaluation splits for
paper comparisons. Training uses pseudo-depth for both backbone and calibration
losses; the GT evaluation loaders use real depth/disparity.

## DSEC: DA2 and LiDAR-aligned disparity

`tools/pseudo_labels/generate_dsec_da2_lidar.py` follows the collaborative
Diffused-Remap generator selected for the DSEC pseudo-label folder. It requires
Python packages in `requirements-event.txt`, a separate Depth Anything V2 source
checkout and its ViT-L weights, and a CUDA GPU (or `--device cpu`). It does not
download models. Check the source and weight licenses for your use.

The DSEC root must contain, for each sequence and matching six-digit frame ID:

```
disparity_maps/SEQUENCE/disparity/event/000000.png
rgb_frames/SEQUENCE/images/left/rectified/000000.png
calib/SEQUENCE/calibration/cam_to_cam.yaml
```

These RGB and disparity frames must already be synchronized and rectified as in
the original preparation; matching names alone does not establish registration.
The script converts event disparity (PNG values divided by 256) to metric depth,
reprojects it to the RGB camera, and robustly fits DA2's relative prediction to
inverse depth. The fitted depth supports reprojection of RGB into the event view.
DA2 then predicts a second time, and the prediction is again fitted to inverse GT
depth in the event view. Finally it inverts and clips depth to 0.5–80 m by
default. This is inverse-depth scale/shift fitting, not quantile matching.

For one frame, from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/pseudo_labels/generate_dsec_da2_lidar.py \
  --dsec-root /path/to/prepared/DSEC/train \
  --da2-repo /path/to/Depth-Anything-V2 \
  --da2-checkpoint /path/to/depth_anything_v2_vitl.pth \
  --output-root /path/to/dsec_pseudo_depth \
  --sequence zurich_city_05_a --frame-index 0
```

Remove `--frame-index` to process all matching frames; repeat `--sequence` for
other sequences, or use `--sequence all`. `--max-frames` limits the global list;
`--num-shards` and `--shard-index` partition it. `--dry-run` prints the selection
without loading the model. Labels are float32 metres, saved as
`SEQUENCE/frame0_pred_depth.npy`, etc. Existing labels are skipped unless
`--overwrite` is specified. The tool stops on an invalid frame; fix the input and
rerun to resume. Missing RGB partners are not selected, so check dataset coverage.

For one DSEC frame the Esharp generator and the original generator produced
identical arrays in the same existing server environment. Four frames were then
regenerated and compared against the label folder referenced by the DSEC splits:
interlaken_00_c frames 0 and 200, and zurich_city_05_a frames 0 and 200. All
arrays were 480 × 640; none was bit-identical. Mean absolute differences were
0.0157, 0.0438, 0.0172 and 0.0123 m, respectively; maximum differences were
6.46, 9.86, 15.53 and 10.55 m. Between 0.014% and 0.204% of pixels differed by
more than 1 m. The cause is unconfirmed, and these checks do not establish
full-dataset or paper-metric reproduction. All 19,217 training and 7,167
validation pseudo-label paths in the release DSEC lists existed on the server
at the time of the check; that does not validate their contents or future paths.

The DSEC alignment helpers were adapted from the collaborative Diffused-Remap
project. The available source checkout did not contain an explicit project
license. Confirm permission, licensing and attribution for these copied helpers
before publishing the repository. Depth Anything V2 code and weights are not
included here.

## MVSEC: teacher inference and quantile alignment

The prepared MVSEC inputs already share an image plane. The pipeline here does
not perform camera reprojection: it runs a teacher on the prepared RGB frame and
fits the result using the valid pixels of the corresponding GT depth map. Match
frame indices and ensure the input data are the intended synchronized and
co-registered preparations. Producing event bins from raw data is separate.

The original workspace contains an MVSEC Depth Anything V2 generator and a
second generator that first fits inverse GT depth to the teacher output with
least squares. The stored intermediate examples tested here are consistent with
that inverse-affine conversion, rather than the raw teacher output. Esharp's
teacher command offers that fit explicitly. The exact full-dataset generation
history and environment remain unverified; do not assume numerical identity.

First generate teacher predictions and fit them to inverse GT depth:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/pseudo_labels/generate_mvsec_teacher.py \
  --mvsec-root /path/to/mvsec/train \
  --inverse-affine-gt-root /path/to/mvsec/train \
  --output-root /path/to/generated_teacher_depth \
  --sequence mvsec_dataset_day2_train --frame-index 0
```

The default teacher is `depth-anything/Depth-Anything-V2-Large-hf`; supply
`--model` for a local Hugging Face model directory and `--local-files-only` to
avoid downloads. Day-2 sequences use `rgb/davis/frame_*.png`; outdoor sequences
use `rgb/davis_left_sync/frame_*.png`. Remove `--frame-index` to process the
sequence, or repeat `--sequence` to add sequences. Without
`--inverse-affine-gt-root`, the saved arrays are raw relative predictions and
are **not** interchangeable with the previously stored intermediate depth maps.
The original generation code is from Event-WassDiff; confirm its license and
permissions, as well as the model's license, before redistribution.

Next, `tools/pseudo_labels/align_mvsec_quantiles.py` aligns the fitted depth
maps to the metric GT depth arrays using quantiles.

The inputs are `TEACHER_ROOT/SEQUENCE/frame_0000000000_pred_depth.npy` and
`GT_ROOT/SEQUENCE/depth/data/depth_0000000000.npy`. The day-2 aliases
`mvsec_dataset_day2_train` and `mvsec_dataset_day2_test` map to
`GT_ROOT/mvsec_dataset_day2/train` and `.../test`, respectively. Predictions
and GT must refer to the same camera view and frame. Arrays are cropped at the
bottom/right to multiples of eight; the resulting shapes must agree.

At jointly valid pixels, the script computes 64 evenly spaced quantiles from
0.05 to 0.95 of the stored teacher values and metric GT **depth**, respectively.
It removes duplicate teacher knots, interpolates monotonically between the
matched knots and clamps values beyond the endpoint knots. Outputs are float32
metres. The defaults require GT between 0.001 m and 80 m and a positive teacher
value, with at least 32 jointly valid pixels. The script rejects non-finite
teacher maps and insufficient or degenerate fits rather than emitting a label.

```bash
python tools/pseudo_labels/align_mvsec_quantiles.py \
  --teacher-root /path/to/generated_teacher_depth \
  --gt-root /path/to/mvsec/train \
  --output-root /path/to/mvsec_pseudo_depth \
  --sequence mvsec_dataset_day2_train --max-frames 1
```

Remove `--max-frames` for the full sequence and repeat `--sequence` as needed.
Existing files are skipped unless `--overwrite` is given. Use the GT root that
corresponds to the intended split; do not generate training labels from test GT.

Four saved labels from the existing data were reproduced exactly from their
stored intermediate arrays with these defaults. This validates those alignment
examples, not the full generation history or all labels. In
particular, these saved examples match quantiles in depth space, not the
inverse-depth quantile formula sometimes used to describe this method; check the
paper description against the actual labels before submission.

In a two-frame end-to-end check using the available cached teacher and an existing
environment (Transformers 4.41.2), the regenerated quantile label differed from
the saved one by a mean of 0.00026 m for outdoor day 1 frame 1 and 0.069 m for
day-2 train frame 1. Maximum absolute differences were 0.044 m and 12.95 m,
respectively. The causes are not confirmed. Transformers 4.41 lacks the newer
processor's depth postprocessing method, so the script uses the original
project's bilinear interpolation fallback on that version. Record the actual
teacher checkpoint, processor and environment to establish exact provenance.

## Remaining release work

Publish or document complete raw-data preparation and event encodings, or provide the prepared inputs
where dataset licenses permit. Record versions, model weights and checksums,
input frame mapping, and final train/evaluation split provenance. Verify the full
reproduction and paper metrics with the released artifacts and environment.
