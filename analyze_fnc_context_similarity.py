import torch
import argparse
import numpy as np
import model.model as module_arch
import data_loader.data_loaders as module_data
from pathlib import Path
from parse_config import ConfigParser
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from utils.util import embed_contexts, embed_fncs


def similarity(config):
    seed = 42
    model_name = 'DSVAE'
    local_size = 4
    context_size = 2

    train_dataloader = config.init_obj('data', module_data, **{'split': 'train', 'shuffle': False})
    valid_dataloader = config.init_obj('data', module_data, **{'split': 'valid', 'shuffle': False})
    test_dataloader = config.init_obj('data', module_data, **{'split': 'test', 'shuffle': False})
    
    extra_args = {
        "input_size": train_dataloader.dataset.data_size,
        "window_size": train_dataloader.dataset.window_size
    }

    np.random.seed(seed)
    torch.manual_seed(seed)
    main_model_name = model_name.lower().replace('conv', '').replace('nln', '')
    experiment_path = Path('saved') / main_model_name / f'models/fBIRNICA/{model_name}_L{local_size}_C{context_size}'
    config_p = experiment_path / f'{seed}-config.json'
    checkpoint_p = experiment_path / f'{seed}-best.pth'
    config = config.load_config(config_p)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    
    model = config.init_obj('arch', module_arch, **extra_args)
    checkpoint = torch.load(checkpoint_p, weights_only=False)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Embed the training, validation and test data into embeddings
    train_subjects = train_dataloader.dataset.num_subjects
    tr_cs, _, _ = embed_contexts(train_dataloader, model, device)
    valid_subjects = valid_dataloader.dataset.num_subjects
    va_cs, _, _ = embed_contexts(valid_dataloader, model, device)
    test_subjects = test_dataloader.dataset.num_subjects
    te_cs, _, _ = embed_contexts(test_dataloader, model, device)
    tr_fnc, _, _ = embed_fncs(train_dataloader)
    va_fnc, _, _ = embed_fncs(valid_dataloader)
    te_fnc, _, _ = embed_fncs(test_dataloader)
    num_subjects = train_subjects + valid_subjects + test_subjects
    num_windows = train_dataloader.dataset.num_windows

    cs = np.concatenate(
        (tr_cs, va_cs, te_cs),
        axis=0
    )
    fncs = np.concatenate(
        (tr_fnc, va_fnc, te_fnc),
        axis=0
    )

    pca = PCA(n_components=context_size)
    fnc_pca = pca.fit_transform(fncs)

    fnc_ssc = StandardScaler(with_mean=True)
    fnc_s = fnc_ssc.fit_transform(fnc_pca)
    cs_ssc = StandardScaler(with_mean=True)
    cs_s = cs_ssc.fit_transform(cs)
    lr = LinearRegression()
    # Direct results
    print(lr.fit(fnc_s, cs_s).score(
        fnc_s, cs_s))

    window_ix_l, window_ix_r = torch.triu_indices(
        num_subjects * num_windows, num_subjects * num_windows, 1)
    
    fncs = torch.from_numpy(fncs).float()
    cs = torch.from_numpy(cs).float()
    # Calculate distances in the feature spaces
    fnc_dist = torch.cdist(fncs, fncs, p=2)[window_ix_l, window_ix_r]
    cs_dist = torch.cdist(cs, cs, p=2)[window_ix_l, window_ix_r]

    # Normalize and then fit a linear model ax + b to predict 
    # the context embedding distances from the FNC distances
    fnc_ssc = StandardScaler(with_mean=True)
    fnc_dist_s = fnc_ssc.fit_transform(fnc_dist[:, np.newaxis])
    cs_ssc = StandardScaler(with_mean=True)
    cs_dist_s = cs_ssc.fit_transform(cs_dist[:, np.newaxis])
    lr = LinearRegression()
    # Distance results
    print(lr.fit(fnc_dist_s, cs_dist_s).score(
        fnc_dist_s, cs_dist_s))

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    args = argparse.ArgumentParser(description='Joint single subject and group neural manifold learning')
    args.add_argument('-rc', '--run_config', default=None, type=str, required=False,
                      help='Run config file path (default: None)')
    args.add_argument('-mc', '--model_config', default=None, type=str, required=False,
                      help='Path to model config (default: None)')
    args.add_argument('-dc', '--data_config', default=None, type=str, required=True,
                      help='Path to data config (default: None)')
    args.add_argument('-hn', '--hyperparameter_index', default=0, type=int,
                      help='Hyperparamter index (default: 0)')
    args = args.parse_args()
    config, hyperparameter_permutations, seeds = ConfigParser.from_args(args)
    similarity(config)
