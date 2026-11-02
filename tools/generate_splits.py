"""Create split lists from prepared files, or convert legacy GT lists.

This tool does not generate event encodings, pseudo-labels, or GT data and does
not choose a paper split. Explicitly select sequences and frame-index mapping.
"""
import argparse
from pathlib import Path
import re


def gt_target(dataset, sequence, index):
    if dataset == 'dsec':
        return Path('event3d/DSEC/train/disparity_maps') / sequence / 'disparity/event' / f'{index:06d}.png'
    prefix = 'mvsec_dataset_day2_'
    sequence = Path('mvsec_dataset_day2') / sequence[len(prefix):] if sequence.startswith(prefix) else Path(sequence)
    return Path('event3d/mvsec/train') / sequence / 'depth/data' / f'depth_{index:010d}.npy'


def convert_legacy(dataset, source):
    pattern = re.compile(r'frame(\d+)_pred_depth\.npy$' if dataset == 'dsec' else r'frame_(\d+)_pred_depth\.npy$')
    result = []
    for lineno, line in enumerate(source.read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f'{source}:{lineno}: expected exactly two columns')
        event, old_target = fields
        match = pattern.search(Path(old_target).name)
        if not match:
            raise ValueError(f'{source}:{lineno}: unrecognized legacy target {old_target}')
        target = gt_target(dataset, Path(old_target).parent.name, int(match.group(1)))
        result.append(f'{event} {target}')
    return result


def pair_files(args):
    result, missing = [], []
    for sequence in args.sequence:
        folder = args.events_root / sequence
        if not folder.is_dir():
            raise FileNotFoundError(f'Event sequence does not exist: {folder}')
        events = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ('.tif', '.tiff'))
        for event in events:
            if not event.stem.isdecimal():
                raise ValueError(f'Event filename must be a numeric frame index: {event}')
            index = int(event.stem) * args.index_multiplier + args.index_offset
            if index < 0:
                raise ValueError(f'Negative target frame index for {event}')
            if args.target == 'gt':
                relative = gt_target(args.dataset, sequence, index)
                target = args.data_root / relative
            else:
                name = f'frame{index}_pred_depth.npy' if args.dataset == 'dsec' else f'frame_{index:010d}_pred_depth.npy'
                target = args.pseudo_root / sequence / name
            if not target.is_file():
                missing.append(str(target))
                continue
            if any(c.isspace() for c in str(event) + str(target)):
                raise ValueError('Split paths cannot contain whitespace')
            result.append(f'{event.resolve()} {target.resolve()}')
    if missing and not args.skip_missing:
        raise FileNotFoundError(f'{len(missing)} matching targets missing; first: {missing[0]}')
    if missing:
        print(f'Skipped {len(missing)} events without matching targets.')
    if not result:
        raise ValueError('No event/target pairs found')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    convert = sub.add_parser('convert-legacy-gt', help='Convert a legacy GT list without changing sample order')
    convert.add_argument('--dataset', required=True, choices=['dsec', 'mvsec'])
    convert.add_argument('--input', required=True, type=Path)
    create = sub.add_parser('pair', help='Pair already prepared events and targets for selected sequences')
    create.add_argument('--dataset', required=True, choices=['dsec', 'mvsec'])
    create.add_argument('--target', required=True, choices=['pseudo', 'gt'])
    create.add_argument('--events-root', required=True, type=Path)
    create.add_argument('--sequence', action='append', required=True, help='Sequence folder; repeat for each sequence')
    create.add_argument('--index-multiplier', required=True, type=int, help='Target index = event index × multiplier + offset')
    create.add_argument('--index-offset', type=int, default=0)
    create.add_argument('--data-root', type=Path, help='Data root containing event3d/ (required for GT)')
    create.add_argument('--pseudo-root', type=Path, help='Pseudo-label root containing sequence folders')
    create.add_argument('--skip-missing', action='store_true', help='Skip unmatched targets instead of failing')
    for p in (convert, create):
        p.add_argument('--output', required=True, type=Path)
        p.add_argument('--overwrite', action='store_true', help='Replace an existing output file')
    args = parser.parse_args()
    if args.command == 'pair' and ((args.target == 'gt' and args.data_root is None) or (args.target == 'pseudo' and args.pseudo_root is None)):
        parser.error('--data-root is required for GT; --pseudo-root is required for pseudo-depth')
    rows = convert_legacy(args.dataset, args.input) if args.command == 'convert-legacy-gt' else pair_files(args)
    with args.output.open('w' if args.overwrite else 'x') as f:
        f.write('\n'.join(rows) + '\n')
    print(f'Wrote {len(rows)} pairs to {args.output}. Confirm sequence and frame alignment before training.')


if __name__ == '__main__':
    main()
