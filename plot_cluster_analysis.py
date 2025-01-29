import torch
import argparse
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import model.model as module_arch
import data_loader.data_loaders as module_data
from pathlib import Path
from sklearn.cluster import KMeans
from parse_config import ConfigParser
from scipy.stats import pearsonr, ttest_ind
from utils.util import embed_contexts, embed_fncs
from scipy.stats import binomtest

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

def cluster_analysis(config):
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
    tr_ge, tr_ix, tr_y = embed_contexts(train_dataloader, model, device)
    va_ge, va_ix, va_y = embed_contexts(valid_dataloader, model, device)
    te_ge, te_ix, te_y = embed_contexts(test_dataloader, model, device)

    # Reshape the contexts into (subjects, windows, latent_dim)
    train_subjects = train_dataloader.dataset.num_subjects
    train_windows = train_dataloader.dataset.num_windows
    tr_ge_time = np.reshape(tr_ge, (train_subjects, train_windows, -1))
    tr_y_subj = np.reshape(tr_y, (train_subjects, train_windows))[:, 0]
    valid_subjects = valid_dataloader.dataset.num_subjects
    valid_windows = valid_dataloader.dataset.num_windows
    va_ge_time = np.reshape(va_ge, (valid_subjects, valid_windows, -1))
    va_y_subj = np.reshape(va_y, (valid_subjects, valid_windows))[:, 0]
    test_subjects = test_dataloader.dataset.num_subjects
    test_windows = test_dataloader.dataset.num_windows
    te_ge_time = np.reshape(te_ge, (test_subjects, test_windows, -1))
    te_y_subj = np.reshape(te_y, (test_subjects, test_windows))[:, 0]

    # Check that the subject_ix is the same for the first 10 windows of subject 1
    print(np.reshape(tr_ix, (train_subjects, train_windows))[0, :10])

    # Create larger arrays of training + validation + test arrays
    context_embeddings_time = np.concatenate((
        tr_ge_time, va_ge_time, te_ge_time
    ), axis=0)
    targets_subj= np.concatenate((
        tr_y_subj, va_y_subj, te_y_subj
    ), axis=0).astype(bool)
    context_embeddings = np.concatenate((
        tr_ge, va_ge, te_ge), axis=0)
    targets = np.concatenate((
        tr_y, va_y, te_y), axis=0).astype(bool)
    df = pd.concat((
        train_dataloader.dataset.df,
        valid_dataloader.dataset.df,
        test_dataloader.dataset.df
    ))

    subject_ixs = np.concatenate((
        tr_ix, va_ix, te_ix
    ), axis=0)

    # Obtain the age of the subjects
    age = df.loc[subject_ixs, 'age'].values
    age = age.astype(np.float32)
    age = (age - age.min()) / (age.max() - age.min())

    # Obtain the CMINDs score for each subjects, and obtain a mask
    # of subjects who do not have CMINDs scores.
    cminds = df.loc[subject_ixs, 'CMINDS_composite'].values
    cminds = cminds.astype(np.float32)
    cminds_notnan = ~np.isnan(cminds) & (cminds != -9999)
    cminds = cminds[cminds_notnan]
    cminds = (cminds - cminds.min()) / (cminds.max() - cminds.min())

    # Create KMeans clusters based on context embeddings from subjects
    # diagnosed with schizophrenia.
    km = KMeans(3, init='k-means++', n_init=500)
    km.fit(context_embeddings[targets])
    print('Cluster locations!')
    for label in range(3):
        print(km.cluster_centers_[label])

    # Define new colormaps
    colors = ["#0071BC", "#93278F"]
    low = mc.to_rgb(colors[0])
    high = mc.to_rgb(colors[1])
    newcolors = np.ones((256, 4))
    for i in range(3):
        newcolors[:128, i] = np.linspace(low[i], 1, 128)
        newcolors[128:, i] = np.linspace(1, high[i], 128)
    cmap = mc.ListedColormap(newcolors)

    # Create a range of plots that show 1) the clusters 2) sz vs c 3) age of sz subjects and 4) CMINDs score of sz subjects
    fig, axs = plt.subplots(1, 4, figsize=(20, 10))
    axs[0].scatter(context_embeddings[targets, 0], context_embeddings[targets, 1], color=colors[1], alpha=0.5, s=20, linewidth=0.)
    axs[0].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='k', s=100, marker='s')
    axs[1].scatter(context_embeddings[targets, 0], context_embeddings[targets, 1], color=colors[1], alpha=0.5, s=20, linewidth=0.)
    axs[1].scatter(context_embeddings[~targets, 0], context_embeddings[~targets, 1], color=colors[0], alpha=0.5, s=20, linewidth=0.)
    axs[2].scatter(context_embeddings[:, 0], context_embeddings[:, 1], c=cmap(age), alpha=0.75, s=20, linewidth=0.)
    axs[3].scatter(context_embeddings[cminds_notnan, 0], context_embeddings[cminds_notnan, 1], c=cmap(cminds), alpha=0.5, s=10)
    axs[0].set_ylim([context_embeddings[:, 1].min() - 0.5, context_embeddings[:, 1].max() + 0.5])
    axs[1].set_ylim([context_embeddings[:, 1].min() - 0.5, context_embeddings[:, 1].max() + 0.5])
    axs[2].set_ylim([context_embeddings[:, 1].min() - 0.5, context_embeddings[:, 1].max() + 0.5])
    axs[3].set_ylim([context_embeddings[:, 1].min() - 0.5, context_embeddings[:, 1].max() + 0.5])
    for i in range(4):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.tight_layout()
    fig.savefig('figures/figure4.png', bbox_inches=0, transparent=False, dpi=500)
    
    # Predict the label for each window from each subject diagnosed
    # with schizophrenia, the number of windows in the train, val and test
    # sets are all the same, so we use train_windows for the reshaping
    labels = km.predict(context_embeddings[targets])
    # (num_sz_subjects, num_windows)
    labels = np.reshape(labels, (int(targets_subj.sum()), train_windows))

    sites = np.concatenate((
            train_dataloader.dataset.df['Site'].values,
            valid_dataloader.dataset.df['Site'].values,
            test_dataloader.dataset.df['Site'].values
        ), axis=0)[targets_subj]
    unique_sites = np.unique(sites)
    print(f'Unique sites: {unique_sites}')
    for label in range(3):
        for site in unique_sites:
            prop_site = (sites == site).sum() / sites.size
            time_in_cluster = (labels == label).mean(-1)
            time_in_cluster_bool = (time_in_cluster > 0.)
            site_in_cluster = ((sites == site) & time_in_cluster_bool).sum()
            num_in_cluster = time_in_cluster_bool.sum()
            result = binomtest(site_in_cluster, num_in_cluster, p=prop_site)
            if result.pvalue < 0.05:
                print('Site significance')
                print(f'Prop in cluster: {site_in_cluster / num_in_cluster}, total prop: {prop_site}, total number: {(sites == site).sum()}')
                print(f'Cluster: {label}, Site: {site}, statistic: {result.statistic}, p-value: {result.pvalue}')

    for col in ['CMINDS_composite', 'age']:
        col_values = np.concatenate((
            train_dataloader.dataset.df[col].values,
            valid_dataloader.dataset.df[col].values,
            test_dataloader.dataset.df[col].values
        ), axis=0)
        # Obtain the non-nan for the column
        col_sz = col_values[targets_subj].copy()
        col_notnan = ~np.isnan(col_sz) & (col_sz != -9999)
        col_nn = col_sz[col_notnan]
        # Loop through each of the three cluster labels
        for label in range(3):
            # Calculate the time a subject spends in a cluster
            time_in_cluster = (labels == label).mean(-1)[col_notnan]
            r, p = pearsonr(time_in_cluster, col_nn)
            if np.abs(p) < 0.05:
                print(f'Cluster: {label}, Column: {col}, pearson')
                print(p, r)
            # Calculate whether a subject spends time in the cluster at all
            time_in_cluster_bool = (time_in_cluster > 0.)
            r, p = ttest_ind(col_nn[time_in_cluster_bool], col_nn[~time_in_cluster_bool])
            if np.abs(p) < 0.05:
                print(f'Cluster: {label}, Column: {col}, ttest')
                print(p, r)

    print('Calculating wFNCs')
    tr_fnc, tr_ix, tr_y = embed_fncs(train_dataloader)
    va_fnc, va_ix, va_y = embed_fncs(valid_dataloader)
    te_fnc, te_ix, te_y = embed_fncs(test_dataloader)

    fncs = np.concatenate((
        tr_fnc, va_fnc, te_fnc), axis=0)
    targets = np.concatenate((
        tr_y, va_y, te_y), axis=0).astype(bool)

    # Only use schizophrenia patients' original input
    fncs_sz = fncs[targets]

    cluster_centers = km.cluster_centers_
    cluster_labels = km.labels_
    # Sort the clusters by the y-axis, since the bottom
    # cluster is the most aligned with control subjects,
    # and the top cluster is most aligned with subjects
    # diagnosed with schizophrenia
    sorted_clusters = np.argsort(cluster_centers[:, 1])
    print(sorted_clusters)

    # Take the average of the FNCs in the cluster and remove schizophrenia mean
    cluster_fncs = []
    for i in range(3):
        cluster_label = sorted_clusters[i]
        cluster_fnc = fncs_sz[cluster_labels == cluster_label].mean(0) - fncs_sz.mean(0)
        cluster_fncs.append(cluster_fnc)

    # Plot each average cluster FNC map
    vminmax = np.max(np.abs(np.array(cluster_fncs)))
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    print(f'Vminmax: {vminmax}')
    r_ix, c_ix = np.triu_indices(53, 1)
    for (i, cluster_fnc) in enumerate(cluster_fncs):
        img = np.zeros((53, 53))
        img[r_ix, c_ix] = cluster_fnc
        img = img + img.T
        axs[i].imshow(img, cmap=cmap, vmin=-vminmax, vmax=vminmax, extent=[0, 53, 0, 53])
        cur_size = 0
        for domain_size in domain_sizes[::-1]:
            cur_size += domain_size
            axs[i].plot([0, 53], [cur_size, cur_size], c='k', linewidth=0.5 ,alpha=0.5)
            axs[i].plot([53-cur_size, 53-cur_size], [53, 0], c='k', linewidth=0.5 ,alpha=0.5)
        axs[i].set_yticks(np.arange(len(comp_names)))
        axs[i].set_yticklabels(comp_names)
        axs[i].tick_params(axis='both', which='major', labelsize=6)
        axs[i].axis('off')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_aspect('equal')
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
    plt.tight_layout()
    plt.savefig('figures/figure5.png', dpi=300, bbox_inches="tight")
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
    cluster_analysis(config)