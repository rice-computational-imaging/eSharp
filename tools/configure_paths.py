"""Resolve release path placeholders in a local checkout, including split files.

Run once after cloning. This is a mechanical text substitution, not a change to
the experiment hyperparameters. Keep the published checkout unconfigured.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--dsec-pseudo-root', type=Path,
                        help='Directory containing sequence folders of DA2/LiDAR pseudo-depth .npy files.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    mapping = {
        '/path/to/repository': str(root),
        '/path/to/data': str(args.data_root.expanduser().resolve()),
        '/path/to/models': str(args.model_root.expanduser().resolve()),
        '/path/to/runs': str(args.run_root.expanduser().resolve()),
    }
    if args.dsec_pseudo_root is not None:
        mapping['/path/to/user/git/Diffused-Remap/outputs/dsec_da2_lidar_pseudo_labels'] = str(
            args.dsec_pseudo_root.expanduser().resolve()
        )
    # Paths are embedded in YAML, Python strings, shell examples and split lists.
    # Restrict characters rather than silently generating invalid source/configs.
    for value in mapping.values():
        if any(c.isspace() or c in "'\"\\$`#" for c in value):
            parser.error('Use paths without spaces, quotes, backslashes or shell metacharacters.')
    changed = 0
    for backend in ('Lotus-ESharp-Fast', 'Marigold-ESharp'):
        for path in sorted((root / backend).rglob('*')):
            if path.suffix not in {'.py', '.yaml', '.yml', '.txt', '.md', '.sh'} or not path.is_file():
                continue
            text = original = path.read_text()
            for old, new in mapping.items():
                text = text.replace(old, new)
            if text != original:
                changed += 1
                if not args.dry_run:
                    path.write_text(text)
    print(f'{"Would configure" if args.dry_run else "Configured"} {changed} files.')
    if args.dsec_pseudo_root is None:
        print('DSEC pseudo-label paths remain unresolved; pass --dsec-pseudo-root to configure them.')
    print('Review stage1.checkpoint and output_dir before training; historical run names are examples.')


if __name__ == '__main__':
    main()
