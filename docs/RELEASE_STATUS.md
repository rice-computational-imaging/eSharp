# Release candidate status

This is a preparation snapshot, not yet the final camera-ready release.

The DSEC pseudo-label generator and MVSEC teacher and quantile steps are now in
`tools/pseudo_labels/`; see [PSEUDO_LABELS.md](PSEUDO_LABELS.md) for limited
sample validation, remaining input preparation, and the permission/license check
required for adapted generation code. Full dataset
generation and paper metrics remain unverified.

## Configuration and calibration update

LPIPS has been removed from all release metric configurations, including
historical snapshots; other metric settings and checkpoint weights are unchanged.
The optional LPIPS implementation remains in the source for explicit use and
is not installed by the root event requirements.

The current Lotus calibration metric L1 reduction now matches the active
Marigold implementation: per-image valid-pixel mean, then batch mean. This is
a training behavior change relative to the original export; saved checkpoint
results have not been recomputed. Both backends retain separate affine-target
and network-specific training logic.

Main DSEC/MVSEC Stage-I, Stage-II and GT evaluation configs are listed in the
root README. First-five and quantile head-only experiment YAMLs were moved out
of this source package. `SOURCE_MANIFEST.json` lists the current source inventory
without checksums. Some historical YAMLs remain for model and
checkpoint provenance; their loader/split choices are not a confirmed paper
protocol. New GT-evaluation configs have not yet undergone full-split validation.

Configs have descriptive dataset/stage names. GT split rows now point directly
to disparity/depth files, and loaders retain compatibility with historical
pseudo-label-style GT rows. The reference split frame order was preserved.
`tools/generate_splits.py` can pair prepared inputs, but it does not generate
the raw representations or define the paper protocol.

The main training configs use pseudo-depth for both backbone and calibration;
the GT evaluation configs use the real-GT loaders. Selected-run metadata for
Lotus DSEC/MVSEC and Marigold DSEC records the pseudo loader. The selected
Marigold MVSEC run's original training log also records the pseudo loader; a
later evaluation invocation recorded GT loaders in the run-level config. This
is why a mutable saved config must not be treated as an immutable training log.

## DSEC smoke checks

Using the existing server environment and one prepared DSEC frame, Lotus-D passed
a calibration training forward/loss check and Marigold passed one training
step. Each backend also loaded a DSEC checkpoint with its calibration head and
evaluated one real-GT frame. The checks used temporary configured copies,
one-frame splits, no LPIPS or visualization, and suppressed Marigold checkpoint
saves. The selected checkpoints for this smoke test are not established as the
paper's final releases. These checks do not verify full splits, MVSEC, a clean
installation, training resume, or numerical paper results.

After direct-GT split conversion and config renaming, all 57,865 DSEC/MVSEC
GT rows were checked for identical ordered event, sequence, and GT frame paths
against their legacy mappings. Real DSEC and MVSEC GT arrays matched for one
legacy/direct pair each. One-frame checkpoint GT inference passed for both
Lotus-D and Marigold on each dataset using temporary configured copies; LPIPS
and visualization were disabled and temporary checkpoint saves suppressed.
These checks used the existing server environment, not a fresh install; they do
not verify the full evaluation splits or reproduce paper metrics.

The author-selected calibrated checkpoints also passed one-frame real-GT
evaluation after adding checkpoint-safety checks. Headless Stage-I DSEC and
MVSEC checkpoints passed one-frame GT evaluation for each backend using the
separate Stage-I configs. CPU regression tests cover headless Stage-I eval,
Stage-II initialization, rejection of a missing calibrated-eval head, both
supported head storage layouts, and evaluation without checkpoint saving.
The recorded MVSEC Marigold `latest` UNet and head were byte-identical to the
selected `iter_008000` files at the time of this check. No full split was run.

The DSEC pseudo-label path setup accepts `--dsec-pseudo-root`, and the event
requirements include xFormers, tabulate and TIFF support. These changes still
need clean-environment installation and end-to-end validation; they are not
evidence of the original paper environment. See DATA.md for data limitations.

## Completed preparation

- Combined current local working-tree source into sibling Lotus-ESharp-Fast/ and
  Marigold-ESharp/ backends.
- Included uncommitted event-training code and configs, not only Git HEAD.
- Excluded nested Git histories, environments, model weights, logs, notebooks,
  image assets and generated training artifacts from this initial source export.
- Replaced known personal account/server paths with neutral placeholders.
- Disabled default Lotus W&B reporting and removed the default team identifier.
- Added path setup instructions, a configuration tool and contributor guidance.
- Preserved upstream licenses and attribution. Backend upstream READMEs are
  historical references; their omitted image/demo assets are not part of this export.

## Required before submission

1. The authors selected four checkpoints and the DSEC/MVSEC GT splits; see
   CHECKPOINTS.md and checkpoint_manifest.json. Verify their full-split metrics
   against the paper and publish checksums and immutable download locations.
2. Confirm whether the local working trees or GitHub/Desktop copies are authoritative.
   The two supplied GitHub URLs were inaccessible without authentication when checked;
   this export is based on the server files, not a verified remote revision.
3. Validate a fresh environment. The dependency snapshot describes the current
   installed environment and may differ from the original experiments.
4. Supply public model download locations and prepared-data/preprocessing instructions.
   Split lists alone are not sufficient to regenerate pseudo-labels. Some auxiliary
   scripts still depend on the separate Event-WassDiff project.
5. Prune auxiliary scripts and stale configs after the official experiments are chosen.
   Search for remaining `/path/to/`, personal identifiers, external absolute paths,
   credentials and nonportable imports before publishing. Known identifier removal
   is not a comprehensive credential audit.
6. Run forward and checkpoint-resume tests for each retained backend, then evaluate
   the selected checkpoints against the paper metrics. No paper results have been
   reproduced as part of this initial export.

## Checks performed during preparation

Checkpoint selection: Lotus-D DSEC step 34,000; Lotus-D MVSEC step 19,500;
Marigold DSEC step 7,000; Marigold MVSEC step 8,000. All selected directories
exist on the preparation server. The Marigold MVSEC 8,000 saved evaluation row
matches the author's eight reported values on the selected GT test split;
the release has not independently rerun that full split. Historical Marigold
iteration numbers were read from `iter_*` marker filenames;
no large pickle-based trainer state was deserialized. Run configs may have been
overwritten by later resumes, so they are metadata evidence, not immutable proof
of the exact configuration at each checkpoint.

- 164 backend Python/YAML files parsed successfully.
- All YAML `base_config` references exist in the combined candidate.
- Lotus-D Stage I and Marigold `--help` imported successfully in the existing
  server environment. These are import checks, not training tests.
- Path configuration dry-run found 44 files requiring local path substitution.
- A text scan for the known original account names, W&B team and server hostname
  found no remaining matches after cleanup (excluding generated bytecode).

Repeat static checks with `python tools/check_source.py` from the repository root.

## Modification notice

The exported backends include event-depth training adaptations to upstream Lotus
and Marigold. Release preparation further replaces machine-specific paths and logging
defaults. SOURCE_MANIFEST.json lists the exported backend files. Preserve upstream
copyright notices; removal of personal runtime settings must not remove attribution.
