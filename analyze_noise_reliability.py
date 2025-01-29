import torch
import argparse
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import model.model as module_arch
import data_loader.data_loaders as module_data
from pathlib import Path
from utils.util import embed_contexts
from parse_config import ConfigParser
from statannotations.Annotator import Annotator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# From: https://stackoverflow.com/questions/35668219/how-to-set-up-a-custom-font-with-custom-path-to-matplotlib-global-font
fe = fm.FontEntry(
    fname='/data/users1/egeenjaar/fonts/montserrat/static/Montserrat-SemiBold.ttf',
    name='Montserrat')
fm.fontManager.ttflist.append(fe)

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['font.family'] = fe.name

def reliability(config):
    train_dataloader = config.init_obj('data', module_data, **{'split': 'train', 'shuffle': False})
    valid_dataloader = config.init_obj('data', module_data, **{'split': 'valid', 'shuffle': False})
    test_dataloader = config.init_obj('data', module_data, **{'split': 'test', 'shuffle': False})

    extra_args = {
        "input_size": train_dataloader.dataset.data_size,
        "window_size": train_dataloader.dataset.window_size
    }

    local_size = 2
    context_size = 2
    seeds = [42, 1337, 1212, 9999]
    models = ['DSVAE', 'IDSVAE']
    noise_types = ['None', 'Low', 'Medium', 'High']

    columns = ['model', 'noise_level', 'result']

    x = "model"
    y = "result"
    order = ['DSVAE', 'IDSVAE']
    colors = ["#93278F", "#0071BC"]
    pairs = [('DSVAE', 'IDSVAE')]

    results_df = pd.DataFrame(
            np.zeros((len(noise_types) * len(models) * len(seeds),
                    len(columns))), columns=columns)
    df_ix = 0
    for (m_ix, model_name) in enumerate(models):
        for (s_ix, seed) in enumerate(seeds):
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

                tr_ge, _, _ = embed_contexts(train_dataloader, model, device)
                va_ge, _, _ = embed_contexts(valid_dataloader, model, device)
                te_ge, _, _ = embed_contexts(test_dataloader, model, device)

                tr_context = np.concatenate((
                    tr_ge, va_ge
                ), axis=0)
                
                tr_base = np.reshape(
                    tr_context, (-1, model.context_size))
                te_base = np.reshape(
                    te_ge, (-1, model.context_size))
                
                ssc_base = StandardScaler()
                tr_base = ssc_base.fit_transform(tr_base)
                te_base = ssc_base.transform(te_base)
                
            for (n_ix, noise_type) in enumerate(noise_types):
                main_model_name = model_name.lower().replace('conv', '').replace('nln', '')
                experiment_path = Path('saved') / main_model_name / f'models/fBIRNICANoise{noise_type}/{model_name}_L{local_size}_C{context_size}'
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

                    tr_ge, _, _ = embed_contexts(train_dataloader, model, device)
                    va_ge, _, _ = embed_contexts(valid_dataloader, model, device)
                    te_ge, _, _ = embed_contexts(test_dataloader, model, device)

                    tr_context = np.concatenate((
                        tr_ge, va_ge
                    ), axis=0)
                    
                    tr_noise = np.reshape(
                        tr_context, (-1, model.context_size))
                    te_noise = np.reshape(
                        te_ge, (-1, model.context_size))

                    ssc_noise = StandardScaler()
                    tr_noise = ssc_noise.fit_transform(tr_noise)
                    te_noise = ssc_noise.transform(te_noise)

                    lr = LinearRegression()
                    lr.fit(tr_base, tr_noise)
                    score = lr.score(te_base, te_noise)
                    print(model_name, noise_type, score)
                    results_df.loc[df_ix, 'model'] = model_name
                    results_df.loc[df_ix, 'noise_level'] = noise_type
                    results_df.loc[df_ix, 'result'] = score
                    df_ix += 1
    
    fig, axs = plt.subplots(1, len(noise_types), figsize=(15, 15))
    for (n_ix, noise_type) in enumerate(noise_types):
        noise_df = results_df[results_df['noise_level'] == noise_type].copy()
        ax = sns.barplot(data=noise_df, x=x, y=y, order=order, ax=axs[n_ix], palette=colors)
        for container in ax.containers:
            for bar in container:
                bar.set_alpha(0.25)
        ax = sns.boxplot(data=noise_df, x=x, y=y, order=order, ax=axs[n_ix], palette=colors)
        axs[n_ix].set_ylim([results_df['result'].min() - 0.05, results_df['result'].max() + 0.05])
        annot = Annotator(ax, pairs, data=noise_df, x=x, y=y, order=order)
        annot.configure(test='t-test_ind', text_format='star', loc='inside')
        annot.apply_test()
        annot.annotate(line_offset=0.01, line_offset_to_group=0.01)
        axs[n_ix].set_title(f'Noise level: {noise_type}')
        if n_ix != 0:
            axs[n_ix].set_ylabel(None)
        axs[n_ix].set_xticklabels([])
        axs[n_ix].set_xlabel(None)
        axs[0].set_ylabel('R-squared')
    plt.tight_layout()
    fig.savefig('figures/figure7.tiff')
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
    reliability(config)
