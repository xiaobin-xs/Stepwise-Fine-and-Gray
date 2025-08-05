import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
from pycox import models
from copy import deepcopy
from lifelines import KaplanMeierFitter
from scipy.interpolate import interp1d
from sklearn.utils.validation import check_is_fitted


def save_finegray(fg_obj, filepath, event, stage=2, exp_id=''):
    # Save torch model state dict
    torch.save(fg_obj.model.state_dict(), filepath + f"stage{stage}_model_{event}_{exp_id}.pt")
    
    # For FineGrayStage2, also save the stage1 model state dict
    # if isinstance(fg_obj, FineGrayStage2):
    #     torch.save(fg_obj.stage1_model.state_dict(), filepath + f"stage{stage}_model_{event}_{exp_id}.pt")
    
    # Save additional attributes using pickle.
    # Make sure that these attributes are picklable.
    extra_state = {
        'ipcw_estimator': fg_obj.ipcw_estimator,
        'base_hz_df': fg_obj.base_hz_df,
        # Include any other parameters that are necessary to reinitialize the object.
    }
    with open(filepath + f"stage{stage}_extra_{event}_{exp_id}.pkl", 'wb') as f:
        pickle.dump(extra_state, f)

def load_finegray(fg_class, net, filepath, event, stage=2, exp_id='', **model_kwargs):
    """
    fg_class: Either FineGray or FineGrayStage2.
    model_constructor: a function or lambda that returns a new instance of the torch model architecture.
    model_kwargs: any additional arguments needed to construct the FineGray object.
    """
    # Create an instance of the FineGray object (or FineGrayStage2) using the provided init_model.
    fg_obj = fg_class(net=net, risk=event, **model_kwargs)
    
    # Load the model state dict
    state_dict = torch.load(filepath + f"stage{stage}_model_{event}_{exp_id}.pt")
    fg_obj.model.load_state_dict(state_dict)
    
    # If FineGrayStage2, load the stage1 model state dict.
    if isinstance(fg_obj, FineGrayStage2):
        stage1_state_dict = torch.load(filepath + f"stage{1}_model_{event}_{exp_id}.pt")
        fg_obj.stage1_model.model.load_state_dict(stage1_state_dict)
    
    # Load the extra attributes
    with open(filepath + f"stage{stage}_extra_{event}_{exp_id}.pkl", 'rb') as f:
        extra_state = pickle.load(f)
    
    fg_obj.ipcw_estimator = extra_state.get('ipcw_estimator', None)
    fg_obj.base_hz_df = extra_state.get('base_hz_df', None)
    
    return fg_obj

# taken from https://github.com/Jeanselme/NeuralFineGray
def get_optimizer(models, lr, optimizer, **kwargs):
    parameters = list(models.parameters())

    if optimizer == 'Adam':
        return torch.optim.Adam(parameters, lr=lr, **kwargs)
    elif optimizer == 'SGD':
        return torch.optim.SGD(parameters, lr=lr, **kwargs)
    elif optimizer == 'RMSProp':
        return torch.optim.RMSprop(parameters, lr=lr, **kwargs)
    else:
        raise NotImplementedError('Optimizer '+optimizer+' is not implemented')
    

# adapted from https://github.com/Jeanselme/NeuralFineGray
def create_representation(inputdim, layers, activation, dropout = 0.):
    if activation == 'ReLU6':
        act = nn.ReLU6()
    elif activation == 'ReLU':
        act = nn.ReLU()
    elif activation == 'Tanh':
        act = nn.Tanh()

    modules = []
    prevdim = inputdim

    for hidden in layers:
        modules.append(nn.Linear(prevdim, hidden, bias = True))
        modules.append(nn.BatchNorm1d(hidden))
        if dropout > 0:
            modules.append(nn.Dropout(p = dropout))
        modules.append(act)
        prevdim = hidden
        
    return modules


# taken from https://github.com/soda-inria/hazardous/blob/main/hazardous/
def check_y_survival(y):
    """Convert DataFrame and dictionnary to record array."""
    y_keys = ["event", "duration"]

    if (
        isinstance(y, np.ndarray)
        and sorted(y.dtype.names, reverse=True) == y_keys
        or isinstance(y, dict)
        and sorted(y, reverse=True) == y_keys
    ):
        return np.ravel(y["event"]), np.ravel(y["duration"])

    elif isinstance(y, pd.DataFrame) and sorted(y.columns, reverse=True) == y_keys:
        return y["event"].values, y["duration"].values

    else:
        raise ValueError(
            "y must be a record array, a pandas DataFrame, or a dict "
            "whose dtypes, keys or columns are 'event' and 'duration'. "
            f"Got:\n{repr(y)}"
        )
    

# taken from https://github.com/soda-inria/hazardous/blob/main/hazardous/_ipcw.py
class KaplanMeierIPCW:
    """Estimate the Inverse Probability of Censoring Weight (IPCW).

    This class estimates the inverse probability of 'survival' to censoring using the
    Kaplan-Meier estimator applied to a binary indicator for censoring, defined as the
    negation of the binary indicator for any event occurrence. This estimator assumes
    that the censoring distribution is independent of the covariates X. If this
    assumption is violated, the estimator may be biased, and a conditional estimator
    might be more appropriate.

    This approach is useful for correcting the bias introduced by right censoring in
    survival analysis, particularly when computing model evaluation metrics such as
    the Brier score or the concordance index.

    Note that the term 'IPCW' can be somewhat misleading: IPCW values represent the
    inverse of the probability of remaining censor-free (or uncensored) at a given time.
    For instance, at t=0, the probability of being censored is 0, so the probability of
    being uncensored is 1.0, and its inverse is also 1.0.

    By construction, IPCW values are always greater than or equal to 1.0 and can only
    increase over time. If no observations are censored, the IPCW values remain
    uniformly at 1.0.

    Note: This estimator extrapolates by maintaining a constant value equal to the last
    observed IPCW value beyond the last recorded time point.

    Parameters
    ----------
    epsilon_censoring_prob : float, default=0.05
        Lower limit of the predicted censoring probabilities. It helps avoiding
        instabilities during the division to obtain IPCW.

    Attributes
    ----------
    min_censoring_prob_ : float
        The effective minimal probability used, defined as the max between
        min_censoring_prob and the minimum predicted probability.

    unique_times_ : ndarray of shape (n_unique_times,)
        The observed censoring durations from the training target.

    censoring_survival_probs_ : ndarray of shape (n_unique_times,)
        The estimated censoring survival probabilities.

    censoring_survival_func_ : callable
        The linear interpolation function defined with unique_times_ (x) and
        censoring_survival_probs_ (y).
    """

    def __init__(self, epsilon_censoring_prob=0.05):
        self.epsilon_censoring_prob = epsilon_censoring_prob

    def fit(self, y, X=None):
        """Marginal estimation of the censoring survival function

        In addition to running the Kaplan-Meier estimator on the negated event
        labels (1 for censoring, 0 for any event), this methods also fits
        interpolation function to be able to make prediction at any time.

        Parameters
        ----------
        y : array-like of shape (n_samples, 2)
            The target data.

        X : None
            The input samples. Unused since this estimator is non-conditional.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        event, duration = check_y_survival(y)
        censoring = event == 0

        km = KaplanMeierFitter()
        km.fit(
            durations=duration,
            event_observed=censoring,
        )

        df = km.survival_function_
        self.unique_times_ = df.index
        self.censoring_survival_probs_ = df.values[:, 0]

        min_censoring_prob = self.censoring_survival_probs_[
            self.censoring_survival_probs_ > 0
        ].min()

        self.min_censoring_prob_ = max(
            min_censoring_prob,
            self.epsilon_censoring_prob,
        )
        self.censoring_survival_func_ = interp1d(
            self.unique_times_,
            self.censoring_survival_probs_,
            kind="previous",
            bounds_error=False,
            fill_value="extrapolate",
        )
        return self

    def compute_ipcw_at(self, times, X=None, ipcw_training=False):
        """Estimate the inverse probability of censoring weights at given time horizons.

        Compute the inverse of the linearly interpolated censoring survival
        function.

        Parameters
        ----------
        times : np.ndarray of shape (n_samples,)
            The input times for which to predict the IPCW for each sample.

        X : None
            The input samples. Unused since this estimator is non-conditional.

        Returns
        -------
        ipcw : np.ndarray of shape (n_samples,)
            The IPCW for each sample at each time horizon.
        """
        check_is_fitted(self, "min_censoring_prob_")

        cs_prob = self.compute_censoring_survival_proba(
            times,
            X=X,
            ipcw_training=ipcw_training,
        )
        cs_prob = np.clip(cs_prob, self.min_censoring_prob_, 1)
        return 1 / cs_prob

    def compute_censoring_survival_proba(self, times, X=None, ipcw_training=False):
        """Estimate probability of not experiencing censoring at times.

        Linearly interpolate the censoring survival function.

        Parameters
        ----------
        times : np.ndarray of shape (n_times,)
            The input times for which to predict the IPCW.

        X : None
            The input samples. Unused since this estimator is non-conditional.

        ipcw_training : bool, default=False
            Unused.

        Returns
        -------
        ipcw : np.ndarray of shape (n_times,)
            The IPCW for times
        """
        return self.censoring_survival_func_(times)


class Net(nn.Module):
    def __init__(self, inputdim, layers=[64, 32], activation='ReLU', dropout=0.2):
        super(Net, self).__init__()
        self.representation = nn.Sequential(*create_representation(inputdim, layers, activation, dropout))
        self.output = nn.Linear(layers[-1], 1, bias = False)
        
    def forward(self, x):
        return self.output(self.representation(x)).view(-1)
    

def neg_partial_loglik_loss(risk_pred, y, e, IPCW, risk=1):
    '''
    compute the negative log- IPCW partial likelihood loss
    nll = - Σ_{i=1}^n w_i(Y_i)δ_i [ f[i] - log( Σ_{j in R_i} w_j(Y_i) exp{f[j]} ) ].
    risk_pred: predicted risk of partial hazard f(x; theta), tensor of shape (n,)
    y: observed/censored times, tensor of shape (n,)
      - e: event indicators (e.g., 0, 1, 2, ...), tensor of shape (n,)
    IPCW: a fitted KaplanMeierIPCW object
    risk: the event (risk) of interest, default is 1
    '''
    n = y.size(0)
    device = y.device

    # -----------------------------
    # 1. Construct the risk set indicator
    # matrix[i, j] = 1 if j is in the risk set of i
    # -----------------------------
    # Natural risk set: j is in R_i if y[j] > y[i]
    natural_mask = y.unsqueeze(0) >= y.unsqueeze(1)  # shape (n, n)
    # Unnatural risk set for indicator (include all observations with an event not equal to risk)
    unnatural_mask = ( ((e != risk) & (e != 0)).unsqueeze(0) ) & (y.unsqueeze(0) < y.unsqueeze(1))
    # Combined risk set indicator (for computing the max of risk_pred in R_i)
    risk_set_indicator_matrix = natural_mask | unnatural_mask
    # TODO: what if the risk set is empty for some observed observations in a mini-batch? 
    # No, it is not possible because the risk set always contains the observation itself.

    # -----------------------------
    # 2. Build the weight matrix for each risk set
    # -----------------------------
    # For natural risk set observations, weight = 1.
    natural_weight = natural_mask.float()
    # For unnatural risk set observations, we restrict to those with y[j] <= y[i] and use a weight ratio.
    # Compute censoring survival probabilities for all y (assumes vectorized operation)
    censoring_surv = torch.tensor(IPCW.compute_censoring_survival_proba(y.cpu()), device=device)  # shape (n,)
    # Create a ratio matrix: ratio[i, j] = censoring_surv[i] / censoring_surv[j]
    ratio_matrix = censoring_surv.unsqueeze(1) / censoring_surv.unsqueeze(0)

    # Compute the weight for the unnatural risk set
    unnatural_weight = ratio_matrix * unnatural_mask.float()

    # Total weight matrix: add the contributions from the two parts
    weight_matrix = natural_weight + unnatural_weight
    # TODO: check if for observations that are not in the risk set, the weights are zeros
        
    # -----------------------------
    # 3. Compute the partial log-likelihood components
    # -----------------------------
    # For numerical stability, compute the maximum risk_pred value over the risk set for each i.
    # Replace entries not in the risk set with -inf so they are ignored in the max.
    risk_pred_masked = torch.where(risk_set_indicator_matrix, 
                                   risk_pred.unsqueeze(0).expand(n, n), 
                                   risk_pred.new_tensor(-float('inf')))
    m = torch.max(risk_pred_masked, dim=1).values  # shape (n,)
    # Compute the denominator sum for each i:
    # sum_j weight_matrix[i, j] * exp(risk_pred[j] - m[i])
    denom = torch.sum(weight_matrix * torch.exp(risk_pred.unsqueeze(0) - m.unsqueeze(1)), dim=1)

    # -----------------------------
    # 4. Compute the negative log-likelihood only for observations with the event of interest.
    # -----------------------------
    risk_mask = e == risk  # Boolean mask for observed risk events # TODO: double-check if w_i(Y_i) is always either 0 or 1
    loss = -torch.sum(risk_pred[risk_mask] - (m[risk_mask] + torch.log(denom[risk_mask])))
    return loss


# largely adapted from https://github.com/Jeanselme/NeuralFineGray/blob/main/nfg/utilities.py#L18
def train_nnfg(model, x_train, t_train, e_train, x_valid, t_valid, e_valid, 
               ipcw_estimator,
               optimizer='Adam', 
               risk=1, n_iter = 100, lr = 1e-4, weight_decay = 0.001,
               bs = 100, patience_max = 20, cuda = False, stage1_model=None, stage1_dim=5):
    '''
    train the FineGray model
    '''

    if cuda:
        model = model.cuda()
    optimizer = get_optimizer(model, lr, optimizer, weight_decay = weight_decay)
    patience, best_loss, previous_loss = 0, np.inf, np.inf
    best_param = deepcopy(model.state_dict())

    if ipcw_estimator is None:
        ipcw_estimator = KaplanMeierIPCW().fit(pd.DataFrame({'event':1* (e_train>0), 'duration': t_train}))

    nbatches = int(x_train.shape[0]/bs) + 1
    index = np.arange(len(x_train))
    t_bar = tqdm(range(n_iter))
    for i in t_bar:
        np.random.shuffle(index)
        model.train()
        
        # Train survival model
        for j in range(nbatches):
            xb = x_train[index[j*bs:(j+1)*bs]]
            tb = t_train[index[j*bs:(j+1)*bs]]
            eb = e_train[index[j*bs:(j+1)*bs]]

            if xb.shape[0] == 0:
                continue

            if cuda:
                xb, tb, eb = xb.cuda(), tb.cuda(), eb.cuda()

            optimizer.zero_grad()
            part_haz_pred = model(xb)
            if stage1_model is not None:
                with torch.no_grad():
                    xb_stage1 = xb[:, :stage1_dim]
                    part_haz_pred_stage1 = stage1_model.predict_partial_hazard_risk(xb_stage1)
                part_haz_pred_stage1 = part_haz_pred_stage1.to(part_haz_pred.device)
                part_haz_pred = part_haz_pred + part_haz_pred_stage1
            loss = neg_partial_loglik_loss(part_haz_pred, tb, eb, ipcw_estimator, risk=risk) 
            loss.backward()
            optimizer.step()

        model.eval()
        xb, tb, eb = x_valid, t_valid, e_valid
        if cuda:
            xb, tb, eb = xb.cuda(), tb.cuda(), eb.cuda()
        with torch.no_grad():
            part_haz_pred = model(xb)
        val_loss = neg_partial_loglik_loss(part_haz_pred, tb, eb, ipcw_estimator, risk=risk)
        t_bar.set_description("Val Loss: {:.3f}".format(val_loss))
        if val_loss < best_loss:
            patience = 0
            best_loss = val_loss
            best_param = deepcopy(model.state_dict())
        elif patience == patience_max:
            break
        else:
            patience += 1

    model.load_state_dict(best_param)
    return model, ipcw_estimator


class FineGray():
    def __init__(self, net, 
                 risk=1, cuda=torch.cuda.is_available(),
                 max_duration=30*24):
        '''
        risk: the risk of interest, default is 1
        '''
        self.model = net
        self.risk = risk
        self.cuda = cuda
        self.fitted = False
        self.base_hz_df = None
        self.max_duration = max_duration
        
    def fit(self, x, y, e, val_data, ipcw_estimator=None,
            optimizer='Adam', **args):
        x_val, y_val, e_val = val_data
        x, y, e = \
            torch.tensor(x, dtype=torch.double), torch.tensor(y, dtype=torch.double), torch.tensor(e, dtype=torch.double)
        x_val, y_val, e_val = \
            torch.tensor(x_val, dtype=torch.double), torch.tensor(y_val, dtype=torch.double), torch.tensor(e_val, dtype=torch.double)
        model, ipcw_estimator = train_nnfg(self.model, x, y, e, x_val, y_val, e_val, ipcw_estimator,
                           optimizer, self.risk, cuda=self.cuda, **args)
        self.model = model.eval()
        self.ipcw_estimator = ipcw_estimator
        self.fitted = True
        self.x = x
        self.y = y
        self.e = e
        return self
        
    def predict_partial_hazard_risk(self, X):
        self.model.eval()
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if self.cuda:
            self.model = self.model.cuda()
        with torch.no_grad():
            if self.cuda:
                X = X.cuda()
            out = self.model(X).cpu()
        if len(X.shape) == 1 or X.shape[0] == 1: # only one subject
            out = out.flatten()
        return out
    
    def calc_base_hazard(self):
        if self.base_hz_df is not None:
            return self.base_hz_df
        
        partial_lnhz = self.predict_partial_hazard_risk(self.x)
        df = pd.DataFrame({'duration_col': self.y, 
                    'event_col': self.e, 
                    'stage_partial_lnhz': partial_lnhz})
        df['expg'] = np.exp(df.stage_partial_lnhz)
        df['censor_surv'] = df.duration_col.apply(
            lambda x: self.ipcw_estimator.compute_censoring_survival_proba(x))
        
        print('Calculating base hazard...')
        t_bar = tqdm(sorted(df.duration_col.unique()))
        t_list = []
        base_hz_list = []
        for t in t_bar:
            if t > self.max_duration:
                break
            t_bar.set_description(f"t: {t:.4f}")
            df_t = df.copy()
            # nominator: number of events of interest occured at time t
            n_events = df_t.loc[(df_t.duration_col == t) & (df_t.event_col == self.risk)].shape[0]

            # natural risk set mask
            natural_mask = df_t.duration_col >= t
            # unnatural risk set mask
            unnatural_mask = (df_t.event_col != self.risk) & (df_t.event_col > 0) & (df_t.duration_col < t)
            
            # weight
            df_t['weight'] = np.inf
            df_t.loc[natural_mask, 'weight'] = 1
            censor_surv_at_t = self.ipcw_estimator.compute_censoring_survival_proba(t)
            df_t.loc[unnatural_mask, 'weight'] = censor_surv_at_t / df_t[unnatural_mask].censor_surv

            # risk set: #patients who are at risk at time t, 
            # including those who have experienced other events
            df_t_at_risk = df_t.loc[natural_mask | unnatural_mask]

            # denominator: sum of weighted partial hazard
            denom = df_t_at_risk.weight * df_t_at_risk.expg

            base_hz = n_events / denom.sum()
            t_list.append(t)
            base_hz_list.append(base_hz)

            

        base_hz_df = pd.DataFrame({'t': t_list, 'base_hz': base_hz_list})
        base_hz_df = base_hz_df.sort_values('t')
        base_hz_df.set_index('t', inplace=True)
        base_hz_df['cumu_base_hz'] = base_hz_df.base_hz.cumsum()

        self.base_hz_df = base_hz_df

        return base_hz_df

    def predict_cif(self, X):
        # TODO: make it accept multiple subjects
        '''
        X: for one subject, tensor of shape (1, n_features) or (n_features,)
        '''
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if self.cuda:
            X = X.cuda()
        if len(X.shape) == 2 and X.shape[0] != 1:
            raise ValueError('X should be only for one subject, your X is of shape {}'.format(X.shape))
        df_cumulative_hazard = self.predict_cumulative_hazard_function(X)
        times = np.array(df_cumulative_hazard.index)
        cum_hz = np.array(df_cumulative_hazard.values.flatten())
        cif = 1 - np.exp(-cum_hz)
        df_cif = pd.DataFrame({'t': times, 'cif': cif})
        df_cif.set_index('t', inplace=True)
        return df_cif
    
    def predict_cumulative_hazard_function(self, X):
        # TODO: make it accept multiple subjects
        '''
        X: for one subject, tensor of shape (1, n_features) or (n_features,)
        '''
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if self.cuda:
            X = X.cuda()
        if len(X.shape) == 2 and X.shape[0] != 1:
            raise ValueError('X should be only for one subject, your X is of shape {}'.format(X.shape))
        if self.base_hz_df is None:
            self.calc_base_hazard()
        partial_loghz = self.predict_partial_hazard_risk(X.view(1, -1))
        expg = torch.exp(partial_loghz).numpy()
        times = np.array(self.base_hz_df.index)
        base_hz = self.base_hz_df.cumu_base_hz.values
        return pd.DataFrame(base_hz.reshape(-1, 1).dot(expg), 
                            index=times)    
    


class FineGrayStage2(FineGray):
    def __init__(self, stage1_model, stage1_dim, net, 
                 risk=1, cuda=torch.cuda.is_available(),
                 max_duration=30*24):
        '''
        stage1_model: the fitted stage 1 model
        stage1_dim: the number of features used in stage 1 model
        net: a init model for stage 2
        risk: the risk of interest, default is 1
        '''
        super().__init__(net, risk, cuda, max_duration)
        self.stage1_model = stage1_model
        self.stage1_dim = stage1_dim

    def fit(self, x, y, e, val_data, ipcw_estimator=None,
            optimizer='Adam', **args):
        x_val, y_val, e_val = val_data
        x, y, e = \
            torch.tensor(x, dtype=torch.double), torch.tensor(y, dtype=torch.double), torch.tensor(e, dtype=torch.double)
        x_val, y_val, e_val = \
            torch.tensor(x_val, dtype=torch.double), torch.tensor(y_val, dtype=torch.double), torch.tensor(e_val, dtype=torch.double)
        model, ipcw_estimator = train_nnfg(self.model, x, y, e, x_val, y_val, e_val, ipcw_estimator, 
                           optimizer, self.risk, 
                           stage1_model=self.stage1_model, stage1_dim=self.stage1_dim, 
                           cuda=self.cuda, **args)
        self.model = model.eval()
        self.ipcw_estimator = ipcw_estimator
        self.fitted = True
        self.x = x
        self.y = y
        self.e = e
        return self
        
    def predict_partial_hazard_risk(self, X, only_stage2=False):
        self.model.eval()
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if self.cuda:
            self.model = self.model.cuda()
        with torch.no_grad():
            if self.cuda:
                X = X.cuda()
            partial_haz_stage1 = self.stage1_model.predict_partial_hazard_risk(X[:, :self.stage1_dim])
            partial_haz_stage2 = self.model(X).cpu().flatten()
        if only_stage2:
            return partial_haz_stage2
        return partial_haz_stage1 + partial_haz_stage2