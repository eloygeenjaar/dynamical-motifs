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

    local_sizes = [2, 4, 8]
    context_sizes = [2, 4, 8]
    seeds = [42, 1337, 1212, 9999]
    models = ['DSVAE', 'IDSVAE']
    results = np.zeros((
        len(local_sizes), len(context_sizes), len(models), (len(seeds) * len(seeds) - len(seeds))
    ))

    columns = ['model', 'result']

    x = "model"
    y = "result"
    order = ['DSVAE', 'IDSVAE']
    colors = ["#93278F", "#0071BC"]
    fig, axs = plt.subplots(len(local_sizes), len(context_sizes), figsize=(15, 15))
    for (l_ix, local_size) in enumerate(local_sizes):
        for (c_ix, context_size) in enumerate(context_sizes):
            results_df = pd.DataFrame(
            np.zeros((len(models) * (len(seeds) * len(seeds) - len(seeds)),
                    len(columns))), columns=columns)
            df_ix = 0
            for (m_ix, model_name) in enumerate(models):
                tr_seeds, te_seeds = [], []
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
                        
                        tr_seeds.append(np.reshape(
                            tr_context, (-1, model.context_size))
                        )
                        te_seeds.append(np.reshape(
                            te_ge, (-1, model.context_size))
                        )

                print(local_size, context_size)
                if len(seeds) == len(te_seeds):
                    seed_results = []
                    for i in range(len(seeds)):
                        for j in range(len(seeds)):
                            if i != j:
                                ssc_i = StandardScaler()
                                ssc_j = StandardScaler()
                                tr_s_i = ssc_i.fit_transform(tr_seeds[i])
                                tr_s_j = ssc_j.fit_transform(tr_seeds[j])
                                te_s_i = ssc_i.transform(te_seeds[i])
                                te_s_j = ssc_j.transform(te_seeds[j])

                                lr = LinearRegression()
                                lr.fit(tr_s_i, tr_s_j)
                                score = lr.score(te_s_i, te_s_j)
                                seed_results.append(score)
                                print(i, j, score)
                                results_df.loc[df_ix, 'model'] = model_name
                                results_df.loc[df_ix, 'result'] = score
                                df_ix += 1
                    results[l_ix, c_ix, m_ix, :] = np.asarray(seed_results)
            pairs = [('DSVAE', 'IDSVAE')]
            ax = sns.barplot(data=results_df, x=x, y=y, order=order, ax=axs[l_ix, c_ix], palette=colors)
            for container in ax.containers:
                for bar in container:
                    bar.set_alpha(0.25)
            ax = sns.boxplot(data=results_df, x=x, y=y, order=order, ax=axs[l_ix, c_ix], palette=colors)
            axs[l_ix, c_ix].set_ylim([results_df['result'].min() - 0.05, results_df['result'].max() + 0.05])
            annot = Annotator(ax, pairs, data=results_df, x=x, y=y, order=order)
            annot.configure(test='t-test_ind', text_format='star', loc='inside')
            annot.apply_test()
            annot.annotate(line_offset=0.01, line_offset_to_group=0.01)
            axs[l_ix, c_ix].set_title(f'LS = {local_size}, CS = {context_size}')
            axs[l_ix, 0].set_ylabel('R-squared')
            if c_ix != 0:
                axs[l_ix, c_ix].set_ylabel(None)
            axs[l_ix, c_ix].set_xticklabels([])
            axs[l_ix, c_ix].set_xlabel(None)
    for (l_ix, local_size) in enumerate(local_sizes):
        for (c_ix, context_size) in enumerate(context_sizes):
            axs[l_ix, c_ix].set_ylim([results.min() - 0.05, results.max() + 0.05])
            if c_ix != 0:
                axs[l_ix, c_ix].set_yticklabels([])
    
    plt.tight_layout()
    fig.savefig('figures/figure3.png', bbox_inches=0, transparent=False, dpi=400)
    plt.clf()
    plt.close(fig)
    print('done')

    

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