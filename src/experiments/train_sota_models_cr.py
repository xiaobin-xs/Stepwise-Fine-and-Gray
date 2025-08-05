"""
====================================
Models: ["deepsurv", 'deephit', 'dsm']
"""
import sys, os
sys.path.append(os.path.abspath('../'))
# 3rd party
import pandas as pd
import numpy as np
import config as cfg
import torch
import random
import warnings
import argparse
import os
from scipy.interpolate import interp1d
from SurvivalEVAL.Evaluator import LifelinesEvaluator

# Local
from utility.survival import (convert_to_structured, make_time_bins, preprocess_data)
from utility.data import dotdict
from utility.config import load_config
from utility.data import (format_data_deephit_competing, format_hierarchical_data_cr, calculate_layer_size_hierarch)
from utility.evaluation import c_index_cr, global_C_index, local_C_index
from ts_data_loader import get_data_loader

# SOTA
from sota_models import (make_cox_model, make_coxboost_model, make_deephit_cr, make_dsm_model, make_rsf_model, train_deepsurv_model,
                         make_deepsurv_prediction, DeepSurv, make_deephit_cr, train_deephit_model)


warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

# Set precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define models
MODELS = ["deepsurv", 'deephit', 'hierarch', 'mtlrcr', 'dsm', 'mensa']
MODELS = ["deepsurv", 'deephit', 'dsm']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_name', type=str, default='bps_cr')
    parser.add_argument('--exclude_covid', type=bool, default=False)
    
    args = parser.parse_args()
    seed = args.seed
    dataset_name = args.dataset_name
    exclude_covid = args.exclude_covid
    for include_bp, include_bpDiff, include_vas, include_cumVas, include_catype in \
        [
            [True, True, True, True, True],
            # [False, False, False, False, True],
            # [True, True, True, True, False],
            # [True, True, False, False, False],
            # [False, False, True, True, False]
            ]:
        feat_include_indicator = np.array([include_bp, include_vas, include_catype]) * 1
        print('include bp, vas, catype:', feat_include_indicator)
        # Load and split data
        dl = get_data_loader(dataset_name)
        dl = dl.load_data(include_bp=include_bp, include_bpDiff=include_bpDiff, 
                          include_vas=include_vas, include_cumVas=include_cumVas, 
                          include_catype=include_catype,
                          exclude_covid=exclude_covid) 
        train_dict, valid_dict, test_dict = dl.split_data(train_size=0.64, valid_size=0.16, test_size=0.2,
                                                        random_state=seed, all_pred_ts=True, k=5)
        n_events = dl.n_events
        
        # Preprocess data
        cat_features = dl.cat_features
        num_features = dl.num_features
        n_features = train_dict['X'].shape[1]
        X_train = pd.DataFrame(train_dict['X'], columns=dl.columns)
        X_valid = pd.DataFrame(valid_dict['X'], columns=dl.columns)
        X_test = pd.DataFrame(test_dict['X'], columns=dl.columns)
        X_train_copy, X_valid_copy = X_train.copy(), X_valid.copy()
        X_train, X_valid, X_test= preprocess_data(X_train, X_valid, X_test, cat_features,
                                                num_features, as_array=True)
        train_dict['X'] = torch.tensor(X_train, device=device, dtype=dtype)
        valid_dict['X'] = torch.tensor(X_valid, device=device, dtype=dtype)
        test_dict['X'] = torch.tensor(X_test, device=device, dtype=dtype)
        for t in [6, 12, 24, 48]:
            X_test = pd.DataFrame(test_dict[f'X_t{t}'], columns=dl.columns)
            _, _, X_test= preprocess_data(X_train_copy, X_valid_copy, X_test, cat_features,
                                                num_features, as_array=True)
            test_dict[f'X_t{t}'] = torch.tensor(X_test, device=device, dtype=dtype)
        n_samples = train_dict['X'].shape[0]
        
        # Make time bins
        time_bins = make_time_bins(train_dict['T'], event=None, dtype=dtype).to(device)
        time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))

        # Evaluate models
        for model_name in MODELS:
            print('*'*12, f'Model: {model_name}', '*'*12)
            # Reset seeds
            np.random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            random.seed(0)
            
            if model_name == "coxph":
                config = dotdict(cfg.COXPH_PARAMS)
                trained_models = []
                for i in range(n_events):
                    train_times = train_dict['T'].cpu().numpy()
                    train_events = (train_dict['E'].cpu().numpy() == i+1)*1.0
                    y_train = convert_to_structured(train_times, train_events)
                    model = make_cox_model(config)
                    model.fit(train_dict['X'].cpu(), y_train)
                    trained_models.append(model)
            elif model_name == "coxboost":
                config = dotdict(cfg.COXBOOST_PARAMS)
                trained_models = []
                for i in range(n_events):
                    train_times = train_dict['T'].cpu().numpy()
                    train_events = (train_dict['E'].cpu().numpy() == i+1)*1.0
                    y_train = convert_to_structured(train_times, train_events)
                    model = make_coxboost_model(config)
                    model.fit(train_dict['X'].cpu(), y_train)
                    trained_models.append(model)
            elif model_name == "rsf":
                config = dotdict(cfg.RSF_PARAMS)
                trained_models = []
                for i in range(n_events):
                    train_times = train_dict['T'].cpu().numpy()
                    train_events = (train_dict['E'].cpu().numpy() == i+1)*1.0
                    y_train = convert_to_structured(train_times, train_events)
                    model = make_rsf_model(config)
                    model.fit(train_dict['X'].cpu(), y_train)
                    trained_models.append(model)
            elif model_name == "deepsurv":
                config = dotdict(cfg.DEEPSURV_PARAMS)
                trained_models = []
                for i in range(n_events):
                    data_train = pd.DataFrame(train_dict['X'].cpu().numpy())
                    data_train['time'] = train_dict['T'].cpu().numpy()
                    data_train['event'] = (train_dict['E'].cpu().numpy() == i+1)*1.0
                    data_valid = pd.DataFrame(valid_dict['X'].cpu().numpy())
                    data_valid['time'] = valid_dict['T'].cpu().numpy()
                    data_valid['event'] = (valid_dict['E'].cpu().numpy() == i+1)*1.0
                    model = DeepSurv(in_features=n_features, config=config)
                    model = train_deepsurv_model(model, data_train, data_valid, time_bins, config=config,
                                                random_state=0, reset_model=True, device=device, dtype=dtype)
                    trained_models.append(model)
            elif model_name == "deephit":
                config = dotdict(cfg.DEEPHIT_PARAMS)
                max_time = torch.tensor([dl.get_data()[1].max()], dtype=dtype, device=device)
                time_bins_dh = time_bins
                if max_time not in time_bins_dh:
                    time_bins_dh = torch.concat([time_bins_dh, max_time], dim=0)
                model = make_deephit_cr(in_features=n_features, out_features=len(time_bins_dh),
                                        num_risks=n_events, duration_index=time_bins_dh, config=config)
                train_data, valid_data, out_features, duration_index = format_data_deephit_competing(train_dict, valid_dict, time_bins_dh)
                model = train_deephit_model(model, train_data['X'], (train_data['T'], train_data['E']),
                                            (valid_data['X'], (valid_data['T'], valid_data['E'])), config)
            elif model_name == "dsm":
                config = dotdict(cfg.DSM_PARAMS)
                n_iter = config['n_iter']
                learning_rate = config['learning_rate']
                batch_size = config['batch_size']
                model = make_dsm_model(config)
                model.fit(train_dict['X'].cpu().numpy(), train_dict['T'].cpu().numpy(), train_dict['E'].cpu().numpy(),
                        val_data=(valid_dict['X'].cpu().numpy(), valid_dict['T'].cpu().numpy(), valid_dict['T'].cpu().numpy()),
                        learning_rate=learning_rate, batch_size=batch_size, iters=n_iter)
            else:
                raise NotImplementedError()
            
            # evaluate model for t = 3, 6, 12, 24
            for t in [6, 12, 24, 48]:
                # Compute survival function
                if model_name in ['coxph', 'coxboost', 'rsf']:
                    all_preds = []
                    for trained_model in trained_models:
                        model_preds = trained_model.predict_survival_function(test_dict[f'X_t{t}'].cpu())
                        model_preds = np.row_stack([fn(time_bins.cpu().numpy()) for fn in model_preds])
                        spline = interp1d(time_bins.cpu().numpy(), model_preds,
                                        kind='linear', fill_value='extrapolate')
                        preds = pd.DataFrame(spline(time_bins.cpu().numpy()),
                                            columns=time_bins.cpu().numpy())
                        all_preds.append(preds)
                elif model_name == "deepsurv":
                    all_preds = []
                    for trained_model in trained_models:
                        preds, time_bins_model = make_deepsurv_prediction(trained_model, test_dict[f'X_t{t}'].to(device),
                                                                        config=config, dtype=dtype)
                        spline = interp1d(time_bins_model.cpu().numpy(),
                                        preds.cpu().numpy(),
                                        kind='linear', fill_value='extrapolate')
                        preds = pd.DataFrame(spline(time_bins.cpu().numpy()),
                                            columns=time_bins.cpu().numpy())
                        all_preds.append(preds)
                elif model_name == "deephit":
                    cif = model.predict_cif(test_dict[f'X_t{t}']).cpu().numpy()
                    all_preds = []
                    for i in range(n_events):
                        preds = pd.DataFrame((1-cif[i]).T, columns=time_bins_dh.cpu().numpy())
                        all_preds.append(preds)
                elif model_name == "dsm":
                    all_preds = []
                    for i in range(n_events):
                        model_pred = model.predict_survival(test_dict[f'X_t{t}'].cpu().numpy(), t=list(time_bins.cpu().numpy()), risk=i+1)
                        model_pred = pd.DataFrame(model_pred, columns=time_bins.cpu().numpy())
                        all_preds.append(model_pred)
                else:
                    raise NotImplementedError()
                
                # Calculate local and global CI
                y_test_time = np.stack([test_dict[f'T_t{t}'].cpu().numpy() for _ in range(n_events)], axis=1)
                y_test_event = np.stack([np.array((test_dict[f'E_t{t}'].cpu().numpy() == i+1)*1.0)
                                        for i in range(n_events)], axis=1)
                all_preds_arr = [df.to_numpy() for df in all_preds]
                global_ci = global_C_index(all_preds_arr, y_test_time, y_test_event)
                local_ci = local_C_index(all_preds_arr, y_test_time, y_test_event)

                # Check for NaN or inf and replace with 0.5
                global_ci = 0.5 if np.isnan(global_ci) or np.isinf(global_ci) else global_ci
                local_ci = 0.5 if np.isnan(local_ci) or np.isinf(local_ci) else local_ci
                
                # Make evaluation for each event
                model_results = pd.DataFrame()
                for event_id, surv_preds in enumerate(all_preds):
                    n_train_samples = len(train_dict['X'])
                    n_test_samples= len(test_dict[f'X_t{t}'])
                    y_train_time = train_dict['T']
                    y_train_event = (train_dict['E'] == event_id+1)*1.0
                    y_test_time = test_dict[f'T_t{t}']
                    y_test_event = (test_dict[f'E_t{t}'] == event_id+1)*1.0
                    y_test_event_cr = test_dict[f'E_t{t}'].cpu().numpy()
                    # find the risk prediction at with column closest to h=240-t
                    h = 240 - t
                    h_idx = np.argmin(np.abs(time_bins.cpu().numpy() - h))
                    surv_preds_at_h = surv_preds.iloc[:, h_idx].values

                    lifelines_eval = LifelinesEvaluator(surv_preds.T, y_test_time, y_test_event,
                                                        y_train_time, y_train_event)
                    cr_ci = c_index_cr(y_test_time, 1-surv_preds_at_h, y_test_event_cr, event_of_interest=event_id+1)
                    ci = lifelines_eval.concordance()[0]
                    ibs = lifelines_eval.integrated_brier_score()
                    mae_hinge = lifelines_eval.mae(method="Hinge")
                    mae_margin = lifelines_eval.mae(method="Margin")
                    mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
                    d_calib = lifelines_eval.d_calibration()[0]
                    
                    metrics = [cr_ci, ci, ibs, mae_hinge, mae_margin, mae_pseudo, d_calib, global_ci, local_ci]
                    print(f'{model_name}: ' + f'{metrics}')
                    res_sr = pd.Series(list(feat_include_indicator) + [model_name, dataset_name, seed, event_id+1] + metrics,
                                        index=['bp', 'vas', 'catype',
                                            "ModelName", "DatasetName", "Seed", "EventId", 
                                            "cr_ci", "CI", "IBS",
                                            "MAEH", "MAEM", "MAEPO", "DCalib", "GlobalCI", "LocalCI"])
                    model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
                    print('event', event_id, cr_ci, ci)
                    
                # Save results
                filename = f"{cfg.RESULTS_DIR}/cota_competing_risks_feats.csv"
                if os.path.exists(filename):
                    results = pd.read_csv(filename)
                else:
                    results = pd.DataFrame(columns=model_results.columns)
                results = results.append(model_results, ignore_index=True)
                results.to_csv(filename, index=False)
            