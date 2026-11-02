"""Check release syntax, YAML, inherited configs, and dataset split references.

This does not load model weights or verify numerical reproduction. Run it in the
event environment so OmegaConf is available.
"""
import ast
from pathlib import Path
from omegaconf import OmegaConf


def main():
    root = Path(__file__).resolve().parents[1]
    count = 0
    missing = []
    for directory in ('tools', 'tests'):
        for path in sorted((root / directory).rglob('*.py')):
            ast.parse(path.read_text(), filename=str(path))
            count += 1
    for backend in ('Lotus-ESharp-Fast', 'Marigold-ESharp'):
        for path in sorted((root / backend).rglob('*')):
            if path.suffix == '.py':
                ast.parse(path.read_text(), filename=str(path))
                count += 1
            elif path.suffix in {'.yaml', '.yml'}:
                cfg = OmegaConf.load(path)
                count += 1
                if OmegaConf.is_dict(cfg):
                    for inherited in cfg.get('base_config', []):
                        if not (root / backend / inherited).is_file():
                            missing.append(f'{path.relative_to(root)} -> {inherited}')
                    if backend == 'Marigold-ESharp':
                        def check_splits(node):
                            if OmegaConf.is_dict(node):
                                for key, value in node.items():
                                    if key == 'filenames' and isinstance(value, str):
                                        split = Path(value)
                                        if not split.is_absolute() and not (root / backend / split).is_file():
                                            missing.append(f'{path.relative_to(root)} -> {value}')
                                    else:
                                        check_splits(value)
                            elif OmegaConf.is_list(node):
                                for item in node:
                                    check_splits(item)
                        check_splits(cfg.get('dataset'))
                    else:
                        data_cfg = OmegaConf.select(cfg, 'data.marigold_config')
                        if data_cfg and not Path(str(data_cfg)).is_absolute():
                            if not (root / 'Marigold-ESharp' / str(data_cfg)).is_file():
                                missing.append(f'{path.relative_to(root)} -> {data_cfg}')
    print(f'Parsed {count} tool/test/backend Python and YAML files.')
    if missing:
        raise SystemExit('Missing configuration or split paths:\n' + '\n'.join(missing))
    print('All inherited configs and relative split/config paths exist.')


if __name__ == '__main__':
    main()
