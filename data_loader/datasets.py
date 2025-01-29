import torch
import numpy as np
import pandas as pd
import nibabel as nb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import datasets
from torch.utils.data import Dataset
from data_loader.utils import get_icafbirn, get_ukbb
from data_loader.definitions import comp_ix
from nilearn import signal
from scipy.interpolate import CubicSpline

class fBIRNICAData(Dataset):
    def __init__(self, data_type: str, window_size: int, window_step: int, fold: int):
        super().__init__()
        self.data_type = data_type
        self.window_size = window_size
        self.window_step = window_step

        # Load the ICA-fBIRN data using a helper function
        train_df, valid_df, test_df = get_icafbirn(42, fold=fold)

        # Use only the dataset corresponding to the data type
        if data_type == 'train':
            self.df = train_df.copy()
        elif data_type == 'valid':
            self.df = valid_df.copy()
        else:
            self.df = test_df.copy()

        self.indices = self.df.index.values

        # Preprocessing the data
        self.data = []
        TR = 2.0
        timesteps = 157
        for (i, row) in self.df.iterrows():
            x = nb.load(row['path']).get_fdata()[:, comp_ix]
            x = signal.clean(
                x,
                t_r=2.0,
                low_pass=0.2,
                high_pass=0.008,
                detrend=True,
                standardize='zscore_sample')
            self.data.append(x)
        self.data = np.stack(self.data, axis=0)
        self.data = torch.from_numpy(self.data).float()
        self.num_subjects, self.num_timesteps, self.data_size = self.data.size()
        # data shape (num_subjects, num_windows, data_size, window_size)
        self.data = self.data.unfold(dimension=1, size=window_size, step=self.window_step)
        self.num_windows = self.data.size(1)
        # Permute to: (num_subjects, num_windows, window_size, data_size)
        self.data = self.data.permute(0, 1, 3, 2)
        self.subject_ixs = torch.arange(self.num_subjects).unsqueeze(1).repeat(1, self.num_windows).view(-1)
        self.window_ixs = torch.arange(self.num_windows).unsqueeze(0).repeat(self.num_subjects, 1).view(-1)
        self.data = torch.reshape(self.data, (-1, self.window_size, self.data_size))

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, ix):
        # Select the data for the subject
        x = self.data[ix]
        subject_ix = self.subject_ixs[ix]
        window_ix = self.window_ixs[ix]
        # Obtain the subject's diagnosis
        # True is schizophrenia
        y = self.df.loc[self.indices[subject_ix], 'sz'] == 1
        return (x, (self.indices[subject_ix], window_ix), y)
    
class fBIRNICADataNoiseLow(fBIRNICAData):

    def __getitem__(self, ix):
        # Select the data for the subject
        x = self.data[ix]
        x = x + torch.randn_like(x) * 0.01
        subject_ix = self.subject_ixs[ix]
        window_ix = self.window_ixs[ix]
        # Obtain the subject's diagnosis
        # True is schizophrenia
        y = self.df.loc[self.indices[subject_ix], 'sz'] == 1
        return (x, (self.indices[subject_ix], window_ix), y)

class fBIRNICADataNoiseMedium(fBIRNICAData):

    def __getitem__(self, ix):
        # Select the data for the subject
        x = self.data[ix]
        x = x + torch.randn_like(x) * 0.05
        subject_ix = self.subject_ixs[ix]
        window_ix = self.window_ixs[ix]
        # Obtain the subject's diagnosis
        # True is schizophrenia
        y = self.df.loc[self.indices[subject_ix], 'sz'] == 1
        return (x, (self.indices[subject_ix], window_ix), y)

class fBIRNICADataNoiseHigh(fBIRNICAData):

    def __getitem__(self, ix):
        # Select the data for the subject
        x = self.data[ix]
        x = x + torch.randn_like(x) * 0.1
        subject_ix = self.subject_ixs[ix]
        window_ix = self.window_ixs[ix]
        # Obtain the subject's diagnosis
        # True is schizophrenia
        y = self.df.loc[self.indices[subject_ix], 'sz'] == 1
        return (x, (self.indices[subject_ix], window_ix), y)

class UKBBICAData(Dataset):
    def __init__(self, data_type: str, window_size: int, window_step: int, fold: int):
        super().__init__()
        self.data_type = data_type
        self.window_size = window_size
        self.window_step = window_step

        # Load the ICA-UKBB data using a helper function
        train_subjects, valid_subjects, test_subjects = get_ukbb(42, fold=fold)

        # Use only the dataset corresponding to the data type
        if data_type == 'train':
            self.subjects = train_subjects
        elif data_type == 'valid':
            self.subjects = valid_subjects
        else:
            self.subjects = test_subjects

        # Preprocessing the data
        TR = 0.74
        new_TR = 2.0
        timesteps = 490
        new_timesteps = int((timesteps * TR) // new_TR)
        old_t = np.linspace(0, (timesteps - 1) * TR, timesteps)
        new_t = np.linspace(0, (new_timesteps - 1) * new_TR, new_timesteps)
        self.data = []
        flag = True
        for (i, subject) in enumerate(self.subjects):
            x = nb.load(subject).get_fdata()[:, comp_ix]
            if x.shape[0] == 490:
                new_x = []
                for i in range(x.shape[-1]):
                    cs = CubicSpline(old_t, x[:, i])
                    new_x.append(cs(new_t))
                if flag:
                    fig, axs = plt.subplots(4, 4)
                    for i in range(4):
                        for j in range(4):
                            ix = i * 4 + j
                            axs[i, j].plot(old_t, x[:, ix])
                            axs[i, j].plot(new_t, new_x[ix])
                    plt.savefig('figures/ukbb_resampling.png')
                    plt.clf()
                    plt.close(fig)
                    flag = False
                x = np.stack(new_x, axis=1)
                x = signal.clean(
                    x,
                    t_r=2.0,
                    low_pass=0.2,
                    high_pass=0.008,
                    detrend=True,
                    standardize='zscore_sample')
                if np.isnan(x).sum() > 0:
                    print(np.isnan(x).sum())
                if np.abs(x).max() > 7:
                    print(x.min(), x.max())
                self.data.append(x)
        self.data = np.stack(self.data, axis=0)
        self.data = torch.from_numpy(self.data).float()
        self.num_subjects, self.num_timesteps, self.data_size = self.data.size()
        # data shape (num_subjects, num_windows, data_size, window_size)
        self.data = self.data.half().unfold(dimension=1, size=window_size, step=self.window_step)
        self.num_windows = self.data.size(1)
        # Permute to: (num_subjects, num_windows, window_size, data_size)
        self.data = self.data.permute(0, 1, 3, 2)
        self.subject_ixs = torch.arange(self.num_subjects).unsqueeze(1).repeat(1, self.num_windows).view(-1)
        self.window_ixs = torch.arange(self.num_windows).unsqueeze(0).repeat(self.num_subjects, 1).view(-1)
        self.data = self.data.contiguous().view(-1, self.window_size, self.data_size)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, ix):
        # Select the data for the subject
        x = self.data[ix].float()
        subject_ix = self.subject_ixs[ix]
        window_ix = self.window_ixs[ix]
        # Obtain the subject's diagnosis
        # True is schizophrenia
        y = torch.zeros((1, )).long()
        return (x, (subject_ix, window_ix), y)
