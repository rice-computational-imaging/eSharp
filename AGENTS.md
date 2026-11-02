# Contributor and agent guide

Read README.md and docs/RELEASE_STATUS.md before changing this repository.

- Lotus-ESharp-Fast/ and Marigold-ESharp/ are sibling backends; Lotus imports Marigold's datasets
  and configuration utilities. Do not rename these directories independently.
- Run each training entry point with its backend directory as the working directory.
- Do not change losses, normalization, masks, splits, learning rates, or schedules
  as part of cosmetic cleanup. These affect experimental results.
- Never commit locally configured paths, credentials, datasets, model weights,
  training states, or experiment logs. Publish the placeholder versions.
- Preserve upstream license, copyright, and citation notices.
- Treat configs as historical experiments until the paper-to-config mapping in
  docs/RELEASE_STATUS.md is completed. Do not claim reproduction from a filename.
- For documentation changes, check paths and commands against the source.
  For code changes, run syntax/import checks, then a relevant forward smoke test
  when models/data/GPU are available. Record what was and was not tested.
- The old upstream environment files are retained for provenance; use the root
  README for the event-training environment instructions.
