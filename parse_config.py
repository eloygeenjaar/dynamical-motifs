import itertools
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json, generate_run_name


class ConfigParser:
    def __init__(self, config):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        """
        # load config file and apply modification
        self._config = config

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])

        exper_name = self.config['name']
        data_name = self.config['data']['type']
        print(config['hyperparameters'])
        run_name = generate_run_name(config['hyperparameters'])
        self._save_dir = save_dir / 'models' / data_name / exper_name / run_name
        self._log_dir = save_dir / 'log' / data_name / exper_name / run_name

        # make directory for saving checkpoints and log.
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        # Load the three main configs
        data_config = read_json(args.data_config)
        if args.run_config is None:
            run_config = read_json('configs/runs/none.json')
        else:
            run_config = read_json(args.run_config)
        if args.model_config is None:
            model_config = read_json('configs/models/none.json')
        else:
            model_config = read_json(args.model_config)

        # From: https://stackoverflow.com/questions/38721847/how-to-generate-all-combination-from-values-in-dict-of-lists-in-python
        hyperparameter_ranges = run_config['hyperparameter_range']
        h_keys, h_vals = zip(*hyperparameter_ranges.items())
        hyperparameter_permutations = [dict(zip(h_keys, v)) for v in itertools.product(*h_vals)]

        # Select the hyperparameters to train with
        if len(hyperparameter_permutations) > 0:
            hyperparameter_dict = hyperparameter_permutations[args.hyperparameter_index]
        else:
            hyperparameter_dict = hyperparameter_permutations
        run_config['hyperparameters'] = dict(hyperparameter_dict)
        run_config['seed'] = hyperparameter_dict['seed']

        # Get the config keys that we want to combine:
        model_keys = set(model_config.keys()).difference(set(data_config.keys())).difference(set(run_config.keys()))
        model_config = {key: model_config[key] for key in model_keys}
        for h_key, h_val in hyperparameter_dict.items():
            if h_key != 'seed':
                model_config['arch']['args'][h_key] = h_val
        
        # Ensure there is no overlap in the keys for each dictionary
        assert len(set(model_config.keys()).intersection(set(data_config.keys())).intersection(set(run_config.keys()))) == 0
        config = {**run_config, **data_config, **model_config}

        return cls(config), hyperparameter_permutations, run_config['hyperparameter_range']['seed']

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    def update_config(self, key, value):
        self._config[key] = value
        
    def update_model(self, old_key, new_key):
        self._config['name'] = self._config['name'].replace(old_key, new_key)
        self._config['arch']['args']['layer_type'] = self._config['arch']['args']['layer_type'].replace(old_key, new_key)
        
    def update_hyperparameters(self, hyperparameters):
        self._config['hyperparameters'] = hyperparameters
        for h_key, h_val in hyperparameters.items():
            if h_key != 'seed':
                self._config['arch']['args'][h_key] = h_val

    @classmethod
    def load_config(cls, p: Path):
        return cls(read_json(p))

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir
