import numpy as np
from torch.utils.data import DataLoader


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.init_kwargs = {
                'dataset': dataset,
                'batch_size': batch_size,
                'num_workers': num_workers,
                'persistent_workers': True,
                'pin_memory': True,
                'prefetch_factor': 4
        }
        self.init_kwargs.update(
            **{'shuffle': shuffle}
        )
        super().__init__(**self.init_kwargs)
