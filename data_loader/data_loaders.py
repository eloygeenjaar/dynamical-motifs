from pathlib import Path
from base import BaseDataLoader
from .datasets import (
    fBIRNICAData, UKBBICAData,
    fBIRNICADataNoiseLow, fBIRNICADataNoiseMedium,
    fBIRNICADataNoiseHigh)

class fBIRNICA(BaseDataLoader):
    def __init__(
        self, split: str, window_size: int, window_step: int, fold: int,
        batch_size: int, num_workers=5, shuffle=True):
        self.dataset = fBIRNICAData(
            data_type=split, window_size=window_size, window_step=window_step, fold=fold)
        super().__init__(
            dataset=self.dataset, batch_size=batch_size if split == 'train' else 512,
            shuffle=shuffle, num_workers=num_workers)

class fBIRNICANoiseLow(BaseDataLoader):
    def __init__(
        self, split: str, window_size: int, window_step: int, fold: int,
        batch_size: int, num_workers=5, shuffle=True):
        self.dataset = fBIRNICADataNoiseLow(
            data_type=split, window_size=window_size, window_step=window_step, fold=fold)
        super().__init__(
            dataset=self.dataset, batch_size=batch_size if split == 'train' else 512,
            shuffle=shuffle, num_workers=num_workers)

class fBIRNICANoiseMedium(BaseDataLoader):
    def __init__(
        self, split: str, window_size: int, window_step: int, fold: int,
        batch_size: int, num_workers=5, shuffle=True):
        self.dataset = fBIRNICADataNoiseMedium(
            data_type=split, window_size=window_size, window_step=window_step, fold=fold)
        super().__init__(
            dataset=self.dataset, batch_size=batch_size if split == 'train' else 512,
            shuffle=shuffle, num_workers=num_workers)

class fBIRNICANoiseHigh(BaseDataLoader):
    def __init__(
        self, split: str, window_size: int, window_step: int, fold: int,
        batch_size: int, num_workers=5, shuffle=True):
        self.dataset = fBIRNICADataNoiseHigh(
            data_type=split, window_size=window_size, window_step=window_step, fold=fold)
        super().__init__(
            dataset=self.dataset, batch_size=batch_size if split == 'train' else 512,
            shuffle=shuffle, num_workers=num_workers)

class UKBBICA(BaseDataLoader):
    def __init__(
        self, split: str, window_size: int, window_step: int, fold: int,
        batch_size: int, num_workers=5, shuffle=True):
        self.dataset = UKBBICAData(
            data_type=split, window_size=window_size, window_step=window_step, fold=fold)
        super().__init__(
            dataset=self.dataset, batch_size=batch_size if split == 'train' else 512,
            shuffle=shuffle, num_workers=num_workers)