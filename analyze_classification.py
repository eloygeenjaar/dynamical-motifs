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
from utils.util import embed_contexts, embed_fncs


def classify(config):
    train_dataloader = config.init_obj('data', module_data, **{'split': 'train', 'shuffle': False})
    valid_dataloader = config.init_obj('data', module_data, **{'split': 'valid', 'shuffle': False})
    test_dataloader = config.init_obj('data', module_data, **{'split': 'test', 'shuffle': False})
    
    extra_args = {
        "input_size": train_dataloader.dataset.data_size,
        "window_size": train_dataloader.dataset.window_size
    }
    seeds = [42, 1337, 9999, 1212]
    models = ['CVAE', 'LVAE', 'ConvDSVAE', 'ConvIDSVAE', 'IDSVAE', 'DSVAE']
    local_sizes = [2, 4, 8, 0]
    context_sizes = [0, 2, 4, 8]
    results_dict = {
        'ConvDSVAE': {'C2_L2': [], 'C4_L2': [], 'C8_L2': []},
        'ConvIDSVAE': {'C2_L2': [], 'C4_L2': [], 'C8_L2': []},
        'DSVAE': {'C2_L2': [], 'C2_L4': [], 'C2_L8': [],
                  'C4_L2': [], 'C4_L4': [], 'C4_L8': [],
                  'C8_L2': [], 'C8_L4': [], 'C8_L8': []},
        'IDSVAE': {'C2_L2': [], 'C2_L4': [], 'C2_L8': [],
                  'C4_L2': [], 'C4_L4': [], 'C4_L8': [],
                  'C8_L2': [], 'C8_L4': [], 'C8_L8': []},
        'LVAE': {'C0_L2': [], 'C0_L4': [], 'C0_L8': []},
        'CVAE': {'C2_L0': [], 'C4_L0': [], 'C8_L0': []},
        'wFNC': {'C2_L0': [], 'C4_L0': [], 'C8_L0': []}
    }
    for (m_ix, model_name) in enumerate(models):
        for (l_ix, local_size) in enumerate(local_sizes):
            for (c_ix, context_size) in enumerate(context_sizes):
                for (s_ix, seed) in enumerate(seeds):
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    main_model_name = model_name.lower().replace('conv', '').replace('nln', '')
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
                        results_dict[model_name][f'C{context_size}_L{local_size}'].append(svm.score(te_ge, te_y))
                
                if f'C{context_size}_L{local_size}' in results_dict[model_name].keys():
                    print(model_name, local_size, context_size, results_dict[model_name][f'C{context_size}_L{local_size}'])
    

    tr_x, _, tr_y = embed_fncs(train_dataloader)
    va_x, _, va_y = embed_fncs(valid_dataloader)
    te_x, _, te_y = embed_fncs(test_dataloader)
    
    tr_x = np.concatenate((tr_x, va_x), axis=0)
    tr_y = np.concatenate((tr_y, va_y), axis=0)

    for context_size in context_sizes[1:]:
        
        pca = PCA(n_components=context_size)
        tr_pca = pca.fit_transform(tr_x)
        te_pca = pca.transform(te_x)
        
        # Scale and then predict on test set
        ssc = StandardScaler()
        tr_pca = ssc.fit_transform(tr_pca)
        te_pca = ssc.transform(te_pca)
        
        svm = SVC(kernel='linear')
        svm.fit(tr_pca, tr_y)
        results_dict['wFNC'][f'C{context_size}_L0'].append(svm.score(te_pca, te_y))
        
        print(context_size, results_dict['wFNC'][f'C{context_size}_L0'])
        del pca

    np.save('results/results.npy', results_dict)

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