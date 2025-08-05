from pathlib import Path

# Directories
ROOT_DIR = Path(__file__).absolute().parent.parent
DATA_DIR = 'data/' #Path.joinpath(ROOT_DIR, "data")
CONFIGS_DIR = Path.joinpath(ROOT_DIR, 'configs')
DATA_CONFIGS_DIR = Path.joinpath(CONFIGS_DIR, 'data')
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')
PLOTS_DIR = Path.joinpath(ROOT_DIR, 'plots')
MODELS_DIR = Path.joinpath(ROOT_DIR, 'models')

# This contains default parameters for the models
DEEPSURV_PARAMS = {
    'hidden_size': 32,
    'verbose': False,
    'lr': 0.0001,
    'c1': 0.01,
    'num_epochs': 1000,
    'dropout': 0.25,
    'batch_size': 128,
    'early_stop': True,
    'patience': 10
}

DSM_PARAMS = {
    'network_layers': [32],
    'learning_rate': 0.001,
    'n_iter': 1000,
    'batch_size': 128
}

DEEPHIT_PARAMS = {
    'num_nodes_shared': [32],
    'num_nodes_indiv': [32],
    'batch_norm': True,
    'verbose': False,
    'dropout': 0.25,
    'alpha': 0.2,
    'sigma': 0.1,
    'batch_size': 128,
    'lr': 0.0001,
    'weight_decay': 0.01,
    'eta_multiplier': 0.8,
    'epochs': 1000,
    'early_stop': True,
    'patience': 10,
}

COXPH_PARAMS = {
    'alpha': 0,
    'ties': 'breslow',
    'n_iter': 100,
    'tol': 1e-9
}