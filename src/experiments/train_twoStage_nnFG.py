import sys, os
sys.path.append(os.path.abspath('../'))
# 3rd party
import pandas as pd
import numpy as np
import config as cfg
import argparse
import pickle
import torch
from lifelines.utils import concordance_index

from nnfg import Net, FineGray, FineGrayStage2, KaplanMeierIPCW, save_finegray, load_finegray
from ts_data_loader import get_data_loader
from utility.survival import make_time_bins
from utility.data import prepare_surv_label_for_sksurv
from utility.data import dotdict
from utility.evaluation import global_C_index, local_C_index, constant_density_interpolate, EvaluatorWithMaxTime


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Set precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = '2nnfg'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_name', type=str, default='bps_cr')
    parser.add_argument('--exclude_covid', type=bool, default=False)
    args = parser.parse_args()
    seed = args.seed
    dataset_name = args.dataset_name
    exclude_covid = args.exclude_covid
    
    include_bp, include_bpDiff, include_vas, include_cumVas, include_catype = [True] * 5
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

    # Make time bins
    time_bins = make_time_bins(train_dict['T'], event=None, dtype=dtype).to(device)
    time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))

    np.random.seed(0)
    if model_name == '2nnfg':
        # config_stage1 = dotdict(cfg.GBSA_PARAMS[0])
        # config_stage2 = dotdict(cfg.GBSA_PARAMS[1]) # TODO: add config for this model
        stage1_features = ['timestep', 
                           'age', 'female', 'transfer', 'oohca', 'cath', 'rhythm_1', 'rhythm_2', 'rhythm_3', 'rhythm_4',
                           'ca_type_1', 'ca_type_2', 'ca_type_3', 'ca_type_4',
                           'FOUR_Motor_0', 'FOUR_Motor_1', 'FOUR_Motor_2', 'FOUR_Motor_3',
                            'FOUR_Motor_4', 'FOUR_Motor_5', 'Pupils_2', 'Pupils_12', 'Corneals_2',
                            'Corneals_12',]
        stage1_models = []
        # only use the first step of each training patient to fit the ipcw estimator
        ipcw_estimator = KaplanMeierIPCW().fit(train_dict['df_label_at_earliest'])
        stage1_all_base_hz, stage2_all_base_hz = [], []
        for event_id in range(1, n_events+1):
            net = Net(inputdim=len(stage1_features))
            stage1_model = FineGray(net, risk=event_id) # NeuralFineGray(layers = param['layers'], layers_surv = param['layers_surv'])
            X_train_stage1 = train_dict['X'][:, :len(stage1_features)]
            X_valid_stage1 = valid_dict['X'][:, :len(stage1_features)]
            stage1_model.fit(X_train_stage1, train_dict['T'].numpy(), train_dict['E'].numpy(), 
                             val_data=(X_valid_stage1, valid_dict['T'].numpy(), valid_dict['E'].numpy()), 
                             ipcw_estimator=ipcw_estimator,
                             n_iter = 1000, bs = 128, lr = 5e-4)
            stage1_models.append(stage1_model)
            stage1_base_hz_df = stage1_models[event_id-1].calc_base_hazard()
            stage1_all_base_hz.append(stage1_base_hz_df)
        for event_id in range(1, n_events+1):
            y_train_time = train_dict['T']
            y_train_event = train_dict['E']
            y_val_time = valid_dict[f'T']
            y_val_event = valid_dict[f'E']
            partial_haz_pred = stage1_models[event_id-1].predict_partial_hazard_risk(torch.tensor(X_valid_stage1))
            ci = concordance_index(y_val_time, -partial_haz_pred.detach().numpy(), 1*(y_val_event==event_id))
            print(f'Event {event_id} C-index: {ci}')
            # TODO: after adding baseline hazard estimation, add iBS
            
        ### Stage 2
        stage2_models = []
        for event_id in range(1, n_events+1):
            net = Net(inputdim=train_dict['X'].shape[1])
            model = FineGrayStage2(stage1_model=stage1_models[event_id-1], stage1_dim=len(stage1_features),
                                   net=net, risk=event_id)
            model.fit(train_dict['X'], train_dict['T'].numpy(), train_dict['E'].numpy(), 
                      val_data=(valid_dict['X'], valid_dict['T'].numpy(), valid_dict['E'].numpy()),
                      ipcw_estimator=ipcw_estimator,
                      n_iter = 1000, bs = 128, lr = 5e-4)
            stage2_models.append(model)
            stage2_base_hz_df = stage2_models[event_id-1].calc_base_hazard()
            stage2_all_base_hz.append(stage2_base_hz_df)
        
        for event_id in range(1, n_events+1):
            y_train_time = train_dict['T']
            y_train_event = train_dict['E']
            y_val_time = valid_dict[f'T']
            y_val_event = valid_dict[f'E']
            
            partial_haz_pred = stage2_models[event_id-1].predict_partial_hazard_risk(torch.tensor(valid_dict['X']))
            ci = concordance_index(y_val_time, -partial_haz_pred.detach().numpy(), 1*(y_val_event==event_id))
            print(f'Event {event_id} C-index: {ci}')

    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    save_object = {'model': model_name,
                    'feat_include': {'include_bp': include_bp, 'include_vas': include_vas, 'include_catype': include_catype},
                    'time_bins': None,
                    'stage1_base_hz_df': stage1_all_base_hz,
                    'stage2_base_hz_df': stage2_all_base_hz,
                    'train': {'pid': train_dict['pid'],
                                'T': train_dict['T'].cpu().numpy(),
                                'E': train_dict['E'].cpu().numpy()},
                    'valid': {},
                    'test': {}}
    for t in [6, 12, 24, 48]:
        valid_pid = valid_dict[f'pid_t{t}']
        if model_name == '2nnfg':
            stage1_all_preds, stage2_all_preds = [], []
            stage1_all_loghr, stage2_all_loghr = [], []
            all_stage1_embeds, all_stage2_embeds = [], []
            stage1_all_cindex, stage1_all_iBS = [], []
            stage2_all_cindex, stage2_all_iBS = [], []
            
            for event_id in range(1, n_events+1):
                y_train_time = train_dict['T']
                y_train_event = train_dict['E']
                y_valid_time = valid_dict[f'T_t{t}']
                y_valid_event = valid_dict[f'E_t{t}']
                ## stage 1
                with torch.no_grad():
                    partial_haz_pred = stage1_models[event_id-1].predict_partial_hazard_risk(torch.tensor(valid_dict[f'X_t{t}'][:, :len(stage1_features)]))
                ci = concordance_index(y_valid_time, -partial_haz_pred.detach().numpy(), 1*(y_valid_event==event_id))
                stage1_all_loghr.append(partial_haz_pred)
                stage1_all_cindex.append(ci)
                

                ## stage 2
                with torch.no_grad():
                    partial_haz_pred = stage2_models[event_id-1].predict_partial_hazard_risk(torch.tensor(valid_dict[f'X_t{t}']), only_stage2=True)
                ci = concordance_index(y_valid_time, -partial_haz_pred.detach().numpy(), 1*(y_valid_event==event_id))
                stage2_all_loghr.append(partial_haz_pred)
                stage2_all_cindex.append(ci)
                
            print(f'Validation set, t={t} C-index: {stage2_all_cindex}')
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        save_object['valid'][f't={t}'] = {'pid': valid_pid,
                                        'T': valid_dict[f'T_t{t}'].cpu().numpy(),
                                        'E': valid_dict[f'E_t{t}'].cpu().numpy(),
                                        'stage1_loghr': stage1_all_loghr,
                                        'stage2_loghr': stage2_all_loghr,
                                        'stage1_cindex': stage1_all_cindex,
                                        'stage2_cindex': stage2_all_cindex,
                                        }
    for t in [6, 12, 24, 48]:
        test_pid = test_dict[f'pid_t{t}']
        if model_name == '2nnfg':
            stage1_all_preds, stage2_all_preds = [], []
            stage1_all_loghr, stage2_all_loghr = [], []
            all_stage1_embeds, all_stage2_embeds = [], []
            stage1_all_cindex, stage1_all_iBS = [], []
            stage2_all_cindex, stage2_all_iBS = [], []
            for event_id in range(1, n_events+1):
                y_train_time = train_dict['T']
                y_train_event = (train_dict['E'] == event_id)*1.0
                y_test_time = test_dict[f'T_t{t}']
                y_test_event = (test_dict[f'E_t{t}'] == event_id)*1.0
                ## stage 1
                with torch.no_grad():
                    partial_haz_pred = stage1_models[event_id-1].predict_partial_hazard_risk(torch.tensor(test_dict[f'X_t{t}'][:, :len(stage1_features)]))
                ci = concordance_index(y_test_time, -partial_haz_pred.detach().numpy(), y_test_event)
                stage1_all_loghr.append(partial_haz_pred)
                stage1_all_cindex.append(ci)

                ## stage 2
                with torch.no_grad():
                    partial_haz_pred = stage2_models[event_id-1].predict_partial_hazard_risk(torch.tensor(test_dict[f'X_t{t}']), only_stage2=True)
                ci = concordance_index(y_test_time, -partial_haz_pred.detach().numpy(), y_test_event)
                stage2_all_loghr.append(partial_haz_pred)
                stage2_all_cindex.append(ci)
            print(f'Test set, t={t} C-index: {stage2_all_cindex}')
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        save_object['test'][f't={t}'] = {'pid': test_pid,
                                        'T': test_dict[f'T_t{t}'].cpu().numpy(),
                                        'E': test_dict[f'E_t{t}'].cpu().numpy(),
                                        'stage1_loghr': stage1_all_loghr,
                                        'stage2_loghr': stage2_all_loghr,
                                        'stage1_cindex': stage1_all_cindex,
                                        'stage2_cindex': stage2_all_cindex}                     
        
    curr_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Save object
    filename = f"{cfg.RESULTS_DIR}/error_analysis/{model_name}_pred_result_{seed}_{curr_time}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(save_object, f)
    print(f'Saved: {filename}')
    # save models
    filepath = f"{cfg.RESULTS_DIR}/models/{model_name}_"
    for e in range(1, n_events+1):
        save_finegray(stage1_models[e-1], filepath, event=e, stage=1, exp_id=f'{seed}_{curr_time}')
        save_finegray(stage2_models[e-1], filepath, event=e, stage=2, exp_id=f'{seed}_{curr_time}')
        # torch.save(stage1_models[e-1].model.state_dict(), f"{cfg.RESULTS_DIR}/models/{model_name}_stage1_model_{e}_{seed}_{curr_time}.pth")
        # torch.save(stage2_models[e-1].model.state_dict(), f"{cfg.RESULTS_DIR}/models/{model_name}_stage2_model_{e}_{seed}_{curr_time}.pth")