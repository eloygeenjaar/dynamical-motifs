import json
import torch
import pandas as pd
from torch import vmap
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from torch.utils.data import Dataset


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, 'total'] += value * n
        self._data.loc[key, 'counts'] += n
        self._data.loc[key, 'average'] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.loc[key, 'average']

    def result(self):
        return dict(self._data.average)

def generate_run_name(hyperparameter_dict):
    run_name = ''
    for hp_name, hp_val in hyperparameter_dict.items():
        if hp_name != 'seed':
            run_name += f'{hp_name}:{hp_val}-'
    run_name += f'seed:{hyperparameter_dict["seed"]}'
    return run_name

def embed_contexts(dataloader, model, device):
    contexts = torch.empty((len(dataloader.dataset), model.context_size), device=device)
    targets = torch.empty((len(dataloader.dataset)))
    subject_ixs = torch.empty((len(dataloader.dataset)), dtype=torch.long)
    start_ix = 0
    for (_, (x, (s_ix, _), y)) in enumerate(dataloader):
        x = x.to(device, non_blocking=True).float()
        with torch.no_grad():
            context_dist = model.embed_context(x)
        end_ix = start_ix + x.size(0)
        contexts[start_ix:end_ix] = context_dist.mean
        targets[start_ix:end_ix] = y
        subject_ixs[start_ix:end_ix] = s_ix
        start_ix = end_ix
    return contexts.cpu().numpy(), subject_ixs.cpu().numpy(), targets.numpy()

def embed_locals(dataloader, model, device):
    local_embeddings = torch.empty((len(dataloader.dataset), dataloader.dataset.window_size, model.local_size), device=device)
    targets = torch.empty((len(dataloader.dataset)))
    subject_ixs = torch.empty((len(dataloader.dataset)), dtype=torch.long)
    start_ix = 0
    for (_, (x, (s_ix, _), y)) in enumerate(dataloader):
        x = x.to(device, non_blocking=True).float()
        with torch.no_grad():
            local_dist = model.embed_local(x)
        end_ix = start_ix + x.size(0)
        local_embeddings[start_ix:end_ix] = local_dist.mean.transpose(1, 0)
        targets[start_ix:end_ix] = y
        subject_ixs[start_ix:end_ix] = s_ix
        start_ix = end_ix
    return local_embeddings.cpu().numpy(), subject_ixs.cpu().numpy(), targets.numpy()

def embed_fncs(dataloader):
    fncs = torch.empty((len(dataloader.dataset), 1378))
    targets = torch.empty((len(dataloader.dataset)))
    subject_ixs = torch.empty((len(dataloader.dataset)), dtype=torch.long)
    row_ix, col_ix = torch.triu_indices(53, 53, offset=1)
    start_ix = 0
    for (_, (x, (s_ix, _), y)) in enumerate(dataloader):
        end_ix = start_ix + x.size(0)
        fncs[start_ix:end_ix] = vmap(torch.corrcoef)(x.transpose(2, 1))[:, row_ix, col_ix]
        targets[start_ix:end_ix] = y
        subject_ixs[start_ix:end_ix] = s_ix
        start_ix = end_ix
    return fncs.cpu().numpy(), subject_ixs.cpu().numpy(), targets.numpy()

def embed_inputs(dataloader):
    inputs = torch.empty((len(dataloader.dataset), dataloader.dataset.window_size, dataloader.dataset.data_size))
    targets = torch.empty((len(dataloader.dataset)))
    subject_ixs = torch.empty((len(dataloader.dataset)), dtype=torch.long)
    start_ix = 0
    for (_, (x, (s_ix, _), y)) in enumerate(dataloader):
        end_ix = start_ix + x.size(0)
        inputs[start_ix:end_ix] = x
        targets[start_ix:end_ix] = y
        subject_ixs[start_ix:end_ix] = s_ix
        start_ix = end_ix
    return inputs.cpu().numpy(), subject_ixs.cpu().numpy(), targets.numpy()
