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

results = np.load('results/results.npy', allow_pickle=True).item()

color_dict = {
    'DSVAE': "#93278F",
    'IDSVAE': "#0071BC",
}

colors = [['#93278F', '#000000'], ['#0071BC', '#000000']]
columns = ['context_size', 'result', 'model']
models = ['DSVAE', 'IDSVAE', 'ConvDSVAE', 'ConvIDSVAE']
hue_order = [['DSVAE', 'ConvDSVAE'], ['IDSVAE', 'ConvIDSVAE']]

context_sizes = [2, 4, 8]
local_size = 2

pairs = [[], []]
for context_size in context_sizes:
    pairs[0].append(((context_size, 'DSVAE'), (context_size, 'ConvDSVAE')))
    pairs[1].append(((context_size, 'IDSVAE'), (context_size, 'ConvIDSVAE')))

df_ls = []
for (m_ix, model_name) in enumerate(models):
    names = list(results[model_name].keys())
    df = pd.DataFrame(np.zeros((
        len(context_sizes) * 4, len(columns)
    )), columns=columns)
    df_ix = 0
    for context_size in context_sizes:
        for s_ix in range(4):
            df.loc[df_ix, 'local_size'] = local_size
            df.loc[df_ix, 'context_size'] = context_size
            df.loc[df_ix, 'model'] = model_name
            print(model_name, context_size)
            if s_ix < len(results[model_name][f'C{context_size}_L{local_size}']):
                df.loc[df_ix, 'result'] = results[model_name][f'C{context_size}_L{local_size}'][s_ix]
            df_ix += 1
    df_ls.append(df)
results_df = pd.concat(df_ls, axis=0)
print(results_df)
print(results_df.loc[results_df['result'].sort_values().index])
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
for (m_ix, model_name) in enumerate(['DSVAE', 'IDSVAE']):
    model_df = results_df[results_df['model'].str.contains(model_name)]
    ax = sns.barplot(data=model_df, x='context_size', y='result', hue='model', hue_order=hue_order[m_ix], ax=axs[m_ix], palette=colors[m_ix])
    for container in ax.containers:
        for bar in container:
            bar.set_alpha(0.25)
    ax = sns.boxplot(data=model_df, x='context_size', y='result', hue='model', hue_order=hue_order[m_ix], ax=axs[m_ix], palette=colors[m_ix])
    axs[m_ix].set_ylim([results_df['result'].min() - 0.02, results_df['result'].max() + 0.05]) 
    annot = Annotator(ax, pairs[m_ix], data=model_df, x='context_size', y='result', hue='model', hue_order=hue_order[m_ix])
    annot.configure(test='t-test_ind', text_format='star', loc='inside')
    annot.apply_test()
    annot.annotate()
    axs[m_ix].legend([model_name, f'Conv{model_name}'], loc='lower right')
    axs[m_ix].set_xlabel('Context size')
    axs[m_ix].set_xticklabels(context_sizes)
    axs[m_ix].set_title(f'{model_name} RNN vs Conv comparison')
    if m_ix == 0:
        axs[m_ix].set_ylabel('Classification accuracy')
        axs[m_ix].set_yticks([0.6, 0.7, 0.8])
        axs[m_ix].set_yticklabels([0.6, 0.7, 0.8])
    else:
        axs[m_ix].set_ylabel(None)
        axs[m_ix].set_yticks([0.6, 0.7, 0.8])
        axs[m_ix].set_yticklabels([0.6, 0.7, 0.8])
    h, l = axs[m_ix].get_legend_handles_labels()
    axs[m_ix].legend(handles=[hi for hi in h[:2]], 
          labels=[li for li in l[:2]], loc='lower right') 
plt.savefig('figures/figure9.tiff')
