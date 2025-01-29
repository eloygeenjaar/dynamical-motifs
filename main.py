import shutil
import argparse
import traceback
import collections
import torch
import numpy as np
import pandas as pd
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from torch.nn import functional as F
from sklearn.model_selection import KFold
from parse_config import ConfigParser
from utils import generate_run_name
from trainer import Trainer
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder


def train(config):
    
    logger = config.get_logger('train')

    # setup data_loader instances
    train_dataloader = config.init_obj('data', module_data, **{'split': 'train', 'shuffle': True})
    valid_dataloader = config.init_obj('data', module_data, **{'split': 'valid', 'shuffle': False})

    extra_args = {
        "input_size": train_dataloader.dataset.data_size,
        "window_size": train_dataloader.dataset.window_size
    }

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch, **extra_args)
    logger.info(model)
    
    # build model architecture, then print to console
    #checkpoint = torch.load('/data/users1/egeenjaar/dynamical-motifs/saved/dsvae/models/UKBBICA/CO/num_layers:1-spatial_hidden_size:128-temporal_hidden_size:256-dropout:0.0-seed:42/model_best.pth')
    #state_dict = checkpoint['state_dict']
    #model.load_state_dict(state_dict)

    # prepare for (multi-device) GPU training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      train_dataloader=train_dataloader,
                      valid_dataloader=valid_dataloader,
                      num_epochs=config['trainer']['epochs'],
                      run_type=config['run_type'])
    trainer.train()

def test(config, checkpoint_p):
    # setup data_loader instances
    train_dataloader = config.init_obj('data', module_data, **{'split': 'valid', 'shuffle': False})
    test_dataloader = config.init_obj('data', module_data, **{'split': 'test', 'shuffle': False})

    extra_args = {
        "input_size": train_dataloader.dataset.data_size,
        "window_size": train_dataloader.dataset.window_size
    }

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch, **extra_args)
    checkpoint = torch.load(checkpoint_p)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # prepare for (multi-device) GPU training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # get function handles of loss and metrics
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    results_dict = {metric.__name__: 0.0 for metric in metrics}
    start_ix = 0
    rec_performance = 0.
    for (_, (x, _, y)) in enumerate(test_dataloader):
        x = x.to(device, non_blocking=True).float()
        with torch.no_grad():
            output = model(x)
        end_ix = start_ix + x.size(0)
        rec_performance = rec_performance + F.mse_loss(output['x_hat'], x) * x.size(0)
        start_ix = end_ix
    print(config['data']['type'])
    results_dict = {'mse': None}
    return results_dict, rec_performance / len(test_dataloader.dataset)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    args = argparse.ArgumentParser(description='Joint single subject and group neural manifold learning')
    args.add_argument('-rc', '--run_config', default=None, type=str, required=True,
                      help='Run config file path (default: None)')
    args.add_argument('-mc', '--model_config', default=None, type=str, required=True,
                      help='Path to model config (default: None)')
    args.add_argument('-dc', '--data_config', default=None, type=str, required=True,
                      help='Path to data config (default: None)')
    args.add_argument('-hn', '--hyperparameter_index', default=0, type=int,
                      help='Hyperparamter index (default: 0)')
    args = args.parse_args()
    config, hyperparameter_permutations, seeds = ConfigParser.from_args(args)
    # fix random seeds for reproducibility
    SEED = config['seed']
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    print(f"NUMBER OF HYPERPARAMETER PERMUTATIONS: {len(hyperparameter_permutations)}")
    flag = True
    valid_df_p = config.save_dir / 'valid.csv'
    if valid_df_p.is_file():
        valid_df = pd.read_csv(valid_df_p, index_col=0)
        flag = (valid_df.loc[config['trainer']['epochs'], 'va_loss'] == 0.0)
    if flag:
        print('training')
        try:
            train(config)
        except Exception as error:
            print(traceback.format_exc())
            # If the run failed, replace the last epochs with nans
            if valid_df_p.is_file():
                valid_df = pd.read_csv(valid_df_p, index_col=0)
                valid_df.loc[valid_df['va_loss'] == 0] = np.nan
                valid_df.to_csv(valid_df_p)

    # Check if all hyperparameters have finished
    experiment_path = (config.save_dir.parent).resolve()
    num_seeds = len(seeds)
    assert len(hyperparameter_permutations) % num_seeds == 0
    all_exist = True
    run_names = []
    for hyperparameter_permutation in hyperparameter_permutations:
        run_name = generate_run_name(hyperparameter_permutation)
        valid_df_p = experiment_path / run_name / 'valid.csv'
        run_names.append(run_name)
        if not valid_df_p.exists():
            all_exist = False
            print(f'Run: {hyperparameter_permutation} not finished')
            print(f'Hyperparameter index: {hyperparameter_permutations.index(hyperparameter_permutation)}')
            print('valid.csv is not found')
            print(valid_df_p)
        else:
            valid_df = pd.read_csv(valid_df_p, index_col=0)
            if valid_df.loc[config['trainer']['epochs'], 'va_loss'] == 0.0:
                all_exist = False
                print(f'Run: {hyperparameter_permutation} not finished')
                print(f'Hyperparameter index: {hyperparameter_permutations.index(hyperparameter_permutation)}')
                print('valid.csv is not finished')
                print(valid_df)
    print(f'All runs exist? {all_exist}')
    print(seeds)
    if all_exist:
        valid_losses = np.zeros((num_seeds, len(hyperparameter_permutations) // num_seeds))
        run_names_noseed = sorted(list(set([run_name.split('-seed:')[0] for run_name in run_names])))
        print(run_names_noseed)
        for hyperparameter_permutation in hyperparameter_permutations:
            run_name = generate_run_name(hyperparameter_permutation)
            run_name_noseed = run_name.split('-seed:')[0]
            valid_df_p = experiment_path / run_name / 'valid.csv'
            valid_df = pd.read_csv(valid_df_p, index_col=0)
            seed_ix = seeds.index(hyperparameter_permutation['seed'])
            run_name_ix = run_names_noseed.index(run_name_noseed)
            valid_losses[seed_ix, run_name_ix] = valid_df['va_loss'][valid_df['va_loss'] != np.nan].min()
        valid_losses_mean = np.mean(valid_losses, axis=0)
        best_hp_ix = np.argmin(valid_losses_mean)
        print(f'Best hyperparameters: {run_names_noseed[best_hp_ix]}')
        print(valid_losses, valid_losses_mean)
        best_hyperparameters = run_names_noseed[best_hp_ix]
        results_dict = {seed: None for seed in seeds}
        for seed in seeds:
            hyperparameter_dict = {}
            for hp_keyval in best_hyperparameters.split('-'):
                key, val = hp_keyval.split(':')
                print(key, val)
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                hyperparameter_dict[key] = val
            hyperparameter_dict['seed'] = seed
            config.update_config('seed', seed)
            config.update_hyperparameters(hyperparameter_dict)
            run_name = generate_run_name(hyperparameter_dict)
            checkpoint_p = experiment_path / run_name / 'model_best.pth'
            print(config['hyperparameters'])
            results_dict[seed], test_rec = test(config, checkpoint_p)
            results_dict[seed]['mse'] = test_rec
        np.save(experiment_path / 'results.npy', results_dict)
        print(results_dict)
        # Remove checkpoints
        for hyperparameter_permutation in hyperparameter_permutations:
            run_name = generate_run_name(hyperparameter_permutation)
            run_name_noseed = run_name.split('-seed:')[0]
            if run_name_noseed != best_hyperparameters:
                last_checkpoint_p = experiment_path / run_name / 'last.pth'
                best_checkpoint_p = experiment_path / run_name / 'model_best.pth'
                last_checkpoint_p.unlink(missing_ok=True)
                best_checkpoint_p.unlink(missing_ok=True)
            else:
                last_checkpoint_p = experiment_path / run_name / 'last.pth'
                best_checkpoint_p = experiment_path / run_name / 'model_best.pth'
                config_p = experiment_path / run_name / 'config.json'
                train_p = experiment_path / run_name / 'train.csv'
                valid_p = experiment_path / run_name / 'valid.csv'
                shutil.copy2(last_checkpoint_p, experiment_path / f'{hyperparameter_permutation["seed"]}-last.pth')
                shutil.copy2(best_checkpoint_p, experiment_path / f'{hyperparameter_permutation["seed"]}-best.pth')
                shutil.copy2(config_p, experiment_path / f'{hyperparameter_permutation["seed"]}-config.json')
                shutil.copy2(train_p, experiment_path / f'{hyperparameter_permutation["seed"]}-train.csv')
                shutil.copy2(valid_p, experiment_path / f'{hyperparameter_permutation["seed"]}-valid.csv')
