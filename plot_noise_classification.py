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

results = np.load('results/noise_results.npy', allow_pickle=True).item()

color_dict = {
    'DSVAE': "#93278F",
    'IDSVAE': "#0071BC"
}

models = ['DSVAE', 'IDSVAE']
colors = ['#93278F', '#0071BC']
x = "noise_level"
y = "result"
colors = ["#93278F", "#0071BC"]
pairs = [('DSVAE', 'IDSVAE')]
noise_types = ['None', 'Low', 'Medium', 'High']
order = noise_types
columns = ['model', 'noise_level', 'result']
pairs = [('None', 'Low'), ('None', 'Medium'), ('None', 'High')]

df = pd.DataFrame(
        np.zeros((len(noise_types) * len(models) * 4,
                len(columns))), columns=columns)

df_ix = 0
for (m_ix, model_name) in enumerate(models):
    names = list(results[model_name].keys())
    for (n_ix, noise_type) in enumerate(noise_types):
        for s_ix in range(4):
            print(results[model_name][noise_type])
            df.loc[df_ix, 'noise_level'] = noise_type
            df.loc[df_ix, 'model'] = model_name
            df.loc[df_ix, 'result'] = results[model_name][noise_type][s_ix]
            df_ix += 1

print(df)
fig, axs = plt.subplots(1, len(models), figsize=(15, 10))
for (m_ix, model_name) in enumerate(models):
    model_df = df[df['model'] == model_name].copy()
    ax = sns.barplot(data=model_df, x=x, y=y, order=order, ax=axs[m_ix], palette=[colors[m_ix]] * len(noise_types))
    for container in ax.containers:
        for bar in container:
            bar.set_alpha(0.25)
    ax = sns.boxplot(data=model_df, x=x, y=y, order=order, ax=axs[m_ix], palette=[colors[m_ix]] * len(noise_types))
    axs[m_ix].set_ylim([df['result'].min() - 0.05, df['result'].max() + 0.05])
    annot = Annotator(ax, pairs, data=model_df, x=x, y=y, order=order)
    annot.configure(test='t-test_ind', text_format='star', loc='inside', line_height=0.005, text_offset=0)
    annot.apply_test()
    annot.annotate(line_offset=0.1, line_offset_to_group=0.1)
    axs[m_ix].set_title(f'Model: {model_name}')
    if m_ix != 0:
        axs[m_ix].set_ylabel(None)
        axs[m_ix].set_yticklabels([])
    axs[m_ix].set_xlabel('Noise level')
    axs[0].set_ylabel('Classification accuracy')
#plt.tight_layout()
fig.savefig('figures/figure6.tiff')
plt.clf()
plt.close(fig)
