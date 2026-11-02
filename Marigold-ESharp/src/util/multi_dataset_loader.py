from typing import List

import torch
from torch.utils.data import DataLoader, Dataset


class CyclicMultiDatasetLoader:
    """
    Iterate through each dataset loader sequentially in one epoch.
    One epoch = full pass over dataset_1, then dataset_2, ... dataset_N.
    """

    def __init__(
        self,
        datasets: List[Dataset],
        batch_size: int,
        num_workers: int,
        shuffle: bool = True,
        generator: torch.Generator = None,
    ) -> None:
        self.datasets = datasets
        self.loaders = [
            DataLoader(
                dataset=ds,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                generator=generator,
            )
            for ds in datasets
        ]
        self.total_length = sum(len(loader) for loader in self.loaders)

    def __len__(self) -> int:
        return self.total_length

    def __iter__(self):
        for dataset_idx, loader in enumerate(self.loaders):
            for batch in loader:
                if isinstance(batch, dict):
                    batch["_dataset_idx"] = dataset_idx
                    batch["_dataset_name"] = getattr(
                        self.datasets[dataset_idx], "disp_name", f"dataset_{dataset_idx}"
                    )
                yield batch
