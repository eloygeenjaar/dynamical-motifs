import torch
import argparse
import importlib
import numpy as np
import model.model as module_arch
import data_loader.data_loaders as module_data
from torch import vmap
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from parse_config import ConfigParser
from utils.util import embed_contexts


def classify(config):
    train_dataloader = config.init_obj('data', module_data, **{'split': 'train', 'shuffle': False})
    valid_dataloader = config.init_obj('data', module_data, **{'split': 'valid', 'shuffle': False})
    test_dataloader = config.init_obj('data', module_data, **{'split': 'test', 'shuffle': False})
    
    extra_args = {
        "input_size": train_dataloader.dataset.data_size,
        "window_size": train_dataloader.dataset.window_size
    }
    seeds = [42, 1337, 9999, 1212]
    models = ['IDSVAE', 'DSVAE']
    local_size = 2
    context_size = 2
    noise_types = ['None', 'Low', 'Medium', 'High']
    results_dict = {
        'DSVAE': {'None': [], 'Low': [], 'Medium': [], 'High': []},
        'IDSVAE': {'None': [], 'Low': [], 'Medium': [], 'High': []}
    }
    for (m_ix, model_name) in enumerate(models):
        for (n_ix, noise_level) in enumerate(noise_types):
            for (s_ix, seed) in enumerate(seeds):
                np.random.seed(seed)
                torch.manual_seed(seed)
                main_model_name = model_name.lower().replace('conv', '').replace('nln', '')
                if noise_level != 'None':
                    experiment_path = Path('saved') / main_model_name / f'models/fBIRNICANoise{noise_level}/{model_name}_L{local_size}_C{context_size}'
                else:
                    experiment_path = Path('saved') / main_model_name / f'models/fBIRNICA/{model_name}_L{local_size}_C{context_size}'
                config_p = experiment_path / f'{seed}-config.json'
                checkpoint_p = experiment_path / f'{seed}-best.pth'
                if config_p.is_file() and checkpoint_p.is_file():
                    config = config.load_config(config_p)
                    device = torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu')
                    
                    model = config.init_obj('arch', module_arch, **extra_args)
                    checkpoint = torch.load(checkpoint_p, weights_only=False)
                    state_dict = checkpoint['state_dict']
                    model.load_state_dict(state_dict)
                    model = model.to(device)
                    model.eval()

                    tr_ge, _, tr_y = embed_contexts(train_dataloader, model, device)
                    va_ge, _, va_y = embed_contexts(valid_dataloader, model, device)
                    te_ge, _, te_y = embed_contexts(test_dataloader, model, device)
                    tr_ge = np.concatenate((tr_ge, va_ge), axis=0)
                    tr_y = np.concatenate((tr_y, va_y), axis=0)

                    # Scale and then predict on test set
                    ssc = StandardScaler()
                    tr_ge = ssc.fit_transform(tr_ge)
                    te_ge = ssc.transform(te_ge)
                    svm = SVC(kernel='linear')
                    svm.fit(tr_ge, tr_y)
                    results_dict[model_name][noise_level].append(svm.score(te_ge, te_y))
            
            if noise_level in results_dict[model_name].keys():
                print(model_name, noise_level, results_dict[model_name][noise_level])

    np.save('results/noise_results.npy', results_dict)

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
    classify(config)