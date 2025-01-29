import torch
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import model.model as module_arch
import data_loader.data_loaders as module_data
from pathlib import Path
from parse_config import ConfigParser
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linear_sum_assignment
from utils.util import embed_contexts, embed_inputs

# From: https://stackoverflow.com/questions/35668219/how-to-set-up-a-custom-font-with-custom-path-to-matplotlib-global-font
fe = fm.FontEntry(
    fname='/data/users1/egeenjaar/fonts/montserrat/static/Montserrat-SemiBold.ttf',
    name='Montserrat')
fm.fontManager.ttflist.append(fe)

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['font.family'] = fe.name

comp_names = [
            'CAU1', 'SUB/HYPOT', 'PUT', 'CAU2', 'THA',

            'STG', 'MTG1',

            'PoCG1', 'L PoCG', 'ParaCL1', 'R PoCG', 'SPL1',
            'ParaCL2', 'PreCG', 'SPL', 'PoCG2',

            'CalcarineG', 'MOG', 'MTG2', 'CUN', 'R MOG',
            'FUG', 'IOG', 'LingualG', 'MTG3',

            'IPL1', 'INS', 'SMFG', 'IFG1', 'R IFG', 'MiFG1',
            'IPL2', 'R IPL', 'SMA', 'SFG', 'MiFG2', 'HiPP1'
            'L IPL', 'MCC', 'IFG2', 'MiFG3', 'HiPP2',

            'Pr1', 'Pr2', 'ACC1', 'PCC1', 'ACC2', 'Pr3', 'PCC2',

            'CB1', 'CB2', 'CB3', 'CB4']

domain_sizes = [5, 2, 9, 9, 17, 7, 4]

def jonker_volgenant(config):
    seed = 42
    model_name = 'DSVAE'
    local_size = 4
    context_size = 2

    valid_dataloader = config.init_obj('data', module_data, **{'split': 'valid', 'shuffle': False})
    
    extra_args = {
        "input_size": valid_dataloader.dataset.data_size,
        "window_size": valid_dataloader.dataset.window_size
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
    va_ge, va_ix, va_y = embed_contexts(valid_dataloader, model, device)
    va_fnc, _, _ = embed_inputs(valid_dataloader)

    # Reshape the contexts into (subjects, windows, latent_dim)
    num_subjects = valid_dataloader.dataset.num_subjects
    num_windows = valid_dataloader.dataset.num_windows
    ge_time = np.reshape(va_ge, (num_subjects, num_windows, -1))
    fnc_time = np.reshape(va_fnc, (num_subjects, num_windows, -1))
    y_subj = np.reshape(va_y, (num_subjects, num_windows))

    side_subjects = int(np.sqrt(num_subjects))
    side_windows = int(np.sqrt(num_windows))
    side = side_subjects * side_windows
    num_subjects = int(side_subjects ** 2)
    num_windows = int(side_windows ** 2)

    x = ge_time[:num_subjects, :num_windows].reshape(num_subjects * num_windows, -1)
    fncs = fnc_time[:num_subjects, :num_windows].reshape(num_subjects * num_windows, -1)
    y = y_subj[:num_subjects, :num_windows].reshape(-1).astype(bool)

    print(x.shape, x.min(0), x.max(0))

    fig, ax = plt.subplots(1, 1)
    ax.scatter(x[y, 0], x[y, 1], c='r')
    ax.scatter(x[~y, 0], x[~y, 1], c='b')
    plt.savefig('figures/figure7_plot.png')
    plt.clf()
    plt.close(fig)

    mms = MinMaxScaler()
    x = mms.fit_transform(x)

    print(side)
    # Create the grid
    _x = np.linspace(0, 1, side)
    _y = np.linspace(0, 1, side)
    xv, yv = np.meshgrid(
            _x,
            _y)
    grid = np.dstack((xv, yv)).reshape(-1, 2)
    # Calculate cost based on embeddings
    cost = cdist(grid, x, "sqeuclidean")
    # See reference why this is necessary:
    # "I've noticed if you normalize to a maximum value that is too large, 
    # this can also cause the Hungarian implementation to crash."
    #cost = cost * (10000000. / cost.max())
    row_ind, col_ind = linear_sum_assignment(cost)
    grid_jv = grid[row_ind]
    fig, axs = plt.subplots(side, side, figsize=(20, 20))
    vminmax = np.abs(fncs).max()
    colors = ['b', 'r']
    r_ix, c_ix = np.triu_indices(53, 1)
    fncs = fncs[col_ind]
    y = y[col_ind]
    for (fnc, ixs, target) in zip(fncs, grid_jv, y.astype(int)):
        img = np.zeros((53, 53))
        img[r_ix, c_ix] = fnc
        img = img + img.T
        ix = (np.abs(_x - ixs[0]).argmin(),
                np.abs(_y - ixs[1]).argmin())
        axs[ix[0], ix[1]].imshow(img, cmap='jet', vmin=-vminmax, vmax=vminmax)
        axs[ix[0], ix[1]].set_xticks([])
        axs[ix[0], ix[1]].set_yticks([])
        #axs[ix[0], ix[1]].set_aspect('equal')
        axs[ix[0], ix[1]].set_xticklabels([])
        axs[ix[0], ix[1]].set_yticklabels([])
        for spine in axs[ix[0], ix[1]].spines.values():
            spine.set_edgecolor(colors[target])
            spine.set_linewidth(1.75)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig('figures/figure7.png', dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close(fig)

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
    jonker_volgenant(config)
