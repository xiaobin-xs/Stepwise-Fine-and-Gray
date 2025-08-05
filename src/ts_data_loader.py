import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List
from pathlib import Path
import config as cfg
import numpy as np
from tqdm import tqdm
from pycop import simulation
from utility.data import kendall_tau_to_theta
from utility.survival import make_stratified_split, make_multi_event_stratified_column
from dgp import DGP_Weibull_linear, DGP_Weibull_nonlinear
import torch
import random
from sklearn import model_selection

# copied from https://github.com/thecml/mensa/blob/main/src/data_loader.py#L16
class BaseDataLoader(ABC):
    """
    Base class for data loaders.
    """
    def __init__(self):
        """Initilizer method that takes a file path, file name,
        settings and optionally a converter"""
        self.X: pd.DataFrame = None
        self.y_t: List[np.ndarray] = None
        self.y_e: List[np.ndarray] = None
        self.num_features: List[str] = None
        self.cat_features: List[str] = None
        self.min_time = None
        self.max_time = None
        self.n_events = None
        self.params = None

    @abstractmethod
    def load_data(self, n_samples) -> None:
        """Loads the data from a data set at startup"""
        
    @abstractmethod
    def split_data(self) -> None:
        """Loads the data from a data set at startup"""

    def get_data(self) -> pd.DataFrame:
        """
        This method returns the features and targets
        :returns: X, y_t and y_e
        """
        return self.X, self.y_t, self.y_e

    def get_features(self) -> List[str]:
        """
        This method returns the names of numerical and categorial features
        :return: the columns of X as a list
        """
        return self.num_features, self.cat_features

    def _get_num_features(self, data) -> List[str]:
        return data.select_dtypes(include=np.number).columns.tolist()

    def _get_cat_features(self, data) -> List[str]:
        return data.select_dtypes(['object']).columns.tolist()

def get_data_loader(dataset_name: str) -> BaseDataLoader:
    if dataset_name in ["bps_cr", "bps_ts_cr"]:
        if dataset_name == "bps_cr":
            return BPSCompetingDataLoader()
        if dataset_name == "bps_ts_cr":
            return BPSTSCompetingDataLoader()
    else:
        raise ValueError("Dataset not found")
    
class BPSCompetingDataLoader(BaseDataLoader): 
    """
    Data loader for Rotterdam dataset (CR).
    """
    def load_data(self, freq: int = 60,
                  n_samples:int = None,
                  include_bp=True, include_bpDiff=True, include_vas=True, include_cumVas=True, include_catype=True,
                  exclude_covid=False):
        '''
        Events: 0 censor, 1 awaken, 2 non-W death, 3 withdrawal
        '''
        df = pd.read_csv(f'{cfg.DATA_DIR}/mbp_vaso_dose_per_min3_2_before_event_occur.csv')
        # exclude FOUR Score - Motor: Follows commands i.e. FOUR_Motor_4 = 1
        df = df[df['FOUR_Motor_4'] != 1]
        print(f"Number of patients: {df['pid'].nunique()}")
        df['mean_bp'] = df['mean_bp'] / 10.0 # scale down to match with the range of vasopressors
        
        df = df.sort_values(by=['pid', 'timestep'])
        df['time_to_event'] = df.time_to_event * 60 # convert hr to min to match the unit of timestep
        # Downsample the data to 1hr frequency
        df = self.downsample_data(df, freq)\
        # rename t_since_arrest_hr to timestep and make to be the nearest integer
        df['timestep'] = df['t_since_arrest_hr']
        df['timestep'] = df['timestep'].round()
        # rename col name
        df = df.rename(columns={'time_to_event': 'time', 'event_indicator': 'event'})
        
        ## change of BP
        df['bp_diff'] = df.groupby('pid')['mean_bp'].diff()
        df['bp_diff'] = df['bp_diff'].fillna(0)
        
        ## cumulative dose of vasopressors
        vas_list = ['DOP_dose', 'EPI_dose', 'NOR_dose', 'VAS_dose', 'PHE_dose']
        cumVas_list = [f'cum_{col}' for col in vas_list]
        cumulative_dose = df.groupby('pid').cumsum()[vas_list]
        cumulative_dose.columns = cumVas_list
        df = pd.concat([df, cumulative_dose], axis=1)
        

        ## feature list
        self.feat_list = self.extract_feat_list(include_bp, include_bpDiff, include_vas, include_cumVas, include_catype)
        
        if n_samples:
            df = df.sample(n=n_samples, random_state=0)
        
        self.original_df = df.copy()
        df = df[df.timestep <= 14400 / freq] # keep at most 10-day data for each patient - computation budget
        if exclude_covid:
            print(f'Before excluding Covid 2020-2021, #unique patients: {df.pid.nunique()}')
            df['year'] = df.pid.apply(lambda x: x[4:8])
            df = df[~(df.year.isin(['2020', '2021']))]
            print(f'After excluding Covid 2020-2021, #unique patients: {df.pid.nunique()}')
        self.df = df
        self.X = df[self.feat_list]
        self.columns = self.feat_list
        self.num_features = self.feat_list
        self.cat_features = []
        self.y_t = df['time']
        self.y_e = df['event']
        self.n_events = 3
        return self

    def downsample_data(self, dataframe, freq):
        dataframe['step'] = dataframe['timestep'] // freq
        dataframe['timestep'] = (1 + dataframe['timestep']) / freq
        dataframe['time_to_event'] = dataframe['time_to_event'] / freq
        dataframe['min_bp'] = dataframe['mean_bp'] # placeholder for min bp
        dataframe['max_bp'] = dataframe['mean_bp'] # placeholder for max bp
        
        dataframe = dataframe.groupby(['pid', 'step']).agg({
            'pid': 'first',
            'mean_bp': 'mean',
            'min_bp': 'min',
            'max_bp': 'max',
            'DOP_dose': 'mean',
            'EPI_dose': 'mean',
            'NOR_dose': 'mean',
            'VAS_dose': 'mean',
            'PHE_dose': 'mean',
            'age': 'last',
            'female': 'last',
            'transfer': 'last',
            'oohca': 'last',
            'cath': 'last',
            'rhythm_1': 'last',
            'rhythm_2': 'last',
            'rhythm_3': 'last',
            'rhythm_4': 'last',
            'ca_type_1': 'last',
            'ca_type_2': 'last',
            'ca_type_3': 'last',
            'ca_type_4': 'last',
            'FOUR_Motor_0': 'last',
            'FOUR_Motor_1': 'last',
            'FOUR_Motor_2': 'last',
            'FOUR_Motor_3': 'last',
            'FOUR_Motor_4': 'last',
            'FOUR_Motor_5': 'last',
            'Pupils_2': 'last',
            'Pupils_12': 'last',
            'Corneals_2': 'last',
            'Corneals_12': 'last',
            'FOUR_Motor': 'last',
            'event_indicator': 'last',
            'time_to_event': 'last',
            'timestep': 'last',
            't_since_arrest_hr': 'last'
        }).reset_index(drop=True)
    
        return dataframe
    
    def extract_feat_list(self, include_bp, include_bpDiff, include_vas, include_cumVas, include_catype):
        ## feature list
        feat_list = ['timestep', 'age', 'female',
                        'transfer', 'oohca', 'cath', 'rhythm_1', 'rhythm_2', 'rhythm_3', 'rhythm_4',
                        'FOUR_Motor_0', 'FOUR_Motor_1', 'FOUR_Motor_2', 'FOUR_Motor_3',
                        'FOUR_Motor_4', 'FOUR_Motor_5', 'Pupils_2', 'Pupils_12', 'Corneals_2',
                        'Corneals_12',] # time elapsed will always be included as a feature
        bp_list = ['mean_bp', 'min_bp', 'max_bp']
        vas_list = ['DOP_dose', 'EPI_dose', 'NOR_dose', 'VAS_dose', 'PHE_dose']
        catype_list = ['ca_type_1', 'ca_type_2', 'ca_type_3', 'ca_type_4']
        if include_catype: # include timestep + ca_type first to make feature slicing in the twoStage training easier
            feat_list += catype_list
        if include_bp:
            feat_list += bp_list
        if include_bpDiff:
            feat_list += ['bp_diff']
        if include_vas:
            feat_list += vas_list
        if include_cumVas:
            cumVas_list = [f'cum_{col}' for col in vas_list]
            feat_list += cumVas_list
        return feat_list
        
    def split_pid(self, seed=42, train_ratio=0.64, val_ratio=0.16, test_ratio=0.20):
        '''
        Stratified split based on event_indicator
        '''
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        pids = self.df['pid'].unique()
        df_pid_event_indicator = self.df[['pid', 'event']].drop_duplicates()
        df_pid_four_motor = self.df[['pid', 'FOUR_Motor']].drop_duplicates()
        event_indicator = [df_pid_event_indicator[df_pid_event_indicator['pid'] == pid]['event'].values[0] for pid in pids]
        # stratify by FOUR score - motor instead of event indicator
        FOUR_Motor = [df_pid_four_motor[df_pid_four_motor['pid'] == pid]['FOUR_Motor'].values[0] for pid in pids]
        # stratified split based on event_indicator
        train_pids, test_pids, train_ei, test_ei = \
            model_selection.train_test_split(pids, event_indicator, test_size=test_ratio, 
                                             random_state=seed, shuffle=True, stratify=FOUR_Motor) #event_indicator)
        train_FOUR_Motor = [df_pid_four_motor[df_pid_four_motor['pid'] == pid]['FOUR_Motor'].values[0] for pid in train_pids]
        train_pids, val_pids, train_ei, val_ei = \
            model_selection.train_test_split(train_pids, train_ei, test_size=val_ratio/(train_ratio+val_ratio), 
                                             random_state=seed, shuffle=True, stratify=train_FOUR_Motor) #train_ei)
        self.train_pids = train_pids
        self.val_pids = val_pids
        self.test_pids = test_pids
        
        return self.train_pids, self.val_pids, self.test_pids
        


    def split_data(self, train_size: float, valid_size: float,
                   test_size: float, dtype=torch.float64, random_state=0, pred_t=6, all_pred_ts=False, k=1):
        '''
        k: int, keep every k hours for the boosted dataset
        '''
        train_pids, val_pids, test_pids = self.split_pid(seed=random_state, train_ratio=train_size, val_ratio=valid_size, test_ratio=test_size)

        # boosted dataset with all time steps
        df_train = self.df[self.df['pid'].isin(train_pids)].reset_index(drop=True)
        df_valid = self.df[self.df['pid'].isin(val_pids)].reset_index(drop=True)
        df_test = self.df[self.df['pid'].isin(test_pids)].reset_index(drop=True)
        # only keep every k hours for the boosted dataset
        # k = 1 # = 5
        df_train = df_train[df_train['timestep'] % k == 0]
        # only keep t=pred_t for val and test sets
        df_valid_t = df_valid[df_valid['timestep'].isin([6., 12., 24., 48.])] # include all pred_time's so that eval is based on average across all pred_t
        df_test_t = df_test[df_test['timestep'] == pred_t]
        print(f"Training set: {df_train.shape[0]} samples")
        dataframes = [df_train, df_valid_t, df_test_t]
        dicts = []
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = dataframe[self.feat_list].to_numpy()
            data_dict['E'] = torch.tensor(dataframe['event'].to_numpy(dtype=np.float64),dtype=dtype)
            data_dict['T'] = torch.tensor(dataframe['time'].to_numpy(dtype=np.float64), dtype=dtype)
            data_dict['pid'] = dataframe['pid'].to_numpy()
            dicts.append(data_dict)

        # add time and event corresponding to the earlist step for each patient in training set
        dicts[0]['df_label_at_earliest'] = df_train.sort_values('timestep').groupby('pid').head(1)[['event', 'time']]\
                                                .rename(columns={'event': 'event', 'time': 'duration'}).reset_index(drop=True)

        if all_pred_ts:
            for t in [6, 12, 24, 48]:
                df_valid_t = df_valid[df_valid['timestep'] == t]
                dicts[1][f'X_t{t}'] = df_valid_t[self.feat_list].to_numpy()
                dicts[1][f'E_t{t}'] = torch.tensor(df_valid_t['event'].to_numpy(dtype=np.float64),dtype=dtype)
                dicts[1][f'T_t{t}'] = torch.tensor(df_valid_t['time'].to_numpy(dtype=np.float64), dtype=dtype)
                dicts[1][f'pid_t{t}'] = df_valid_t['pid'].to_numpy()
                df_test_t = df_test[df_test['timestep'] == t]
                dicts[2][f'X_t{t}'] = df_test_t[self.feat_list].to_numpy()
                dicts[2][f'E_t{t}'] = torch.tensor(df_test_t['event'].to_numpy(dtype=np.float64),dtype=dtype)
                dicts[2][f'T_t{t}'] = torch.tensor(df_test_t['time'].to_numpy(dtype=np.float64), dtype=dtype)
                dicts[2][f'pid_t{t}'] = df_test_t['pid'].to_numpy()
            
        return dicts[0], dicts[1], dicts[2]
    
    def get_data_one_pid(self, pid=None, dtype=torch.float64):
        if pid is None:
            pid = random.choice(self.test_pids)
        df = self.original_df[self.original_df['pid'] == pid].reset_index(drop=True)
        df = df.sort_values('timestep', ascending=True)
        data_dict = dict()
        data_dict['X'] = df[self.feat_list].to_numpy()
        data_dict['E'] = torch.tensor(df['event'].to_numpy(dtype=np.float64),dtype=dtype)
        data_dict['T'] = torch.tensor(df['time'].to_numpy(dtype=np.float64), dtype=dtype)
        data_dict['t_since_arrest_hr'] = df['t_since_arrest_hr']
        data_dict['pid'] = pid
        return data_dict