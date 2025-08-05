'''
Many of the code taken from the NeuralFineGray repository: https://github.com/Jeanselme/NeuralFineGray with minor changes;
I write the part to make the 2stage trainig work
'''
# import sys
# sys.path.append('../../NeuralFineGray/DeepSurvivalMachines/')

from dsm.dsm_api import DSMBase
from tqdm import tqdm
from torch.autograd import grad
import numpy as np
from copy import deepcopy
import torch.nn as nn
import torch

def total_loss_cs(model, x, t, e):
  # Go through network
  log_sr, _, tau = model.forward(x, t)
  log_hr = model.gradient(log_sr, tau, e).log()

  # Likelihood error
  error = 0
  for k in range(model.risks):
    error -= log_sr[e != (k + 1)][:, k].sum()
    error -= log_hr[e == (k + 1)].sum()

  return error / len(x)


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
   

def train_nfg(model, total_loss,
        x_train, t_train, e_train,
        x_valid, t_valid, e_valid,
        n_iter = 1000, lr = 1e-3, weight_decay = 0.001,
        bs = 100, patience_max = 3, cuda = False):
  # ToDo: add eval using c-index / brier score
  # Separate oprimizer as one might need more time to converge
  optimizer = get_optimizer(model, lr, model.optimizer, weight_decay = weight_decay)

  patience, best_loss, previous_loss = 0, np.inf, np.inf
  best_param = deepcopy(model.state_dict())

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
      loss = total_loss(model, xb, tb, eb) 
      loss.backward()
      optimizer.step()

    model.eval()
    xb, tb, eb = x_valid, t_valid, e_valid
    if cuda:
      xb, tb, eb  = xb.cuda(), tb.cuda(), eb.cuda()

    valid_loss = total_loss(model, xb, tb, eb).item() 
    t_bar.set_description("Loss: {:.3f}".format(valid_loss))
    if valid_loss < previous_loss:
      patience = 0

      if valid_loss < best_loss:
        best_loss = valid_loss
        best_param = deepcopy(model.state_dict())

    elif patience == patience_max:
      break
    else:
      patience += 1

    previous_loss = valid_loss

  model.load_state_dict(best_param)
  return model, i


# All of this as dependence
class PositiveLinear(nn.Module):
  def __init__(self, in_features, out_features, bias = False):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
    if bias:
      self.bias = nn.Parameter(torch.Tensor(out_features))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.xavier_uniform_(self.log_weight)
    if self.bias is not None:
      fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.log_weight)
      bound = np.sqrt(1 / np.sqrt(fan_in))
      nn.init.uniform_(self.bias, -bound, bound)
    self.log_weight.data.abs_().sqrt_()

  def forward(self, input):
    if self.bias is not None:
      return nn.functional.linear(input, self.log_weight ** 2, self.bias)
    else:
      return nn.functional.linear(input, self.log_weight ** 2)


def create_representation_positive(inputdim, layers, dropout = 0):
  modules = []
  act = nn.Tanh()

  prevdim = inputdim
  for hidden in layers:
    modules.append(PositiveLinear(prevdim, hidden, bias = True))
    if dropout > 0:
      modules.append(nn.Dropout(p = dropout))
    modules.append(act)
    prevdim = hidden

  # Need all values positive
  modules[-1] = nn.Softplus()

  return nn.Sequential(*modules)


def create_representation(inputdim, layers, activation, dropout = 0., last = None):
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

  if not(last is None):
    modules[-1] = last
  
  return modules


class NeuralFineGrayTorch(nn.Module):

  def __init__(self, inputdim, embed_dim = 32, layers = [64], act = 'ReLU', layers_surv = [64],
               risks = 1, dropout = 0.3, optimizer = "Adam", multihead = True):
    super().__init__()
    self.input_dim = inputdim
    self.risks = risks  # Competing risks
    self.dropout = dropout
    self.optimizer = optimizer

    self.embed = nn.Sequential(*create_representation(inputdim, layers + [embed_dim], act, self.dropout)) # Assign each point to a cluster
    self.balance = nn.Sequential(*create_representation(embed_dim, layers + [risks], act)) # Define balance between outcome (ensure sum < 1)
    self.outcome = nn.ModuleList(
                      [create_representation_positive(embed_dim + 1, layers_surv + [1]) # Multihead (one for each outcome)
                  for _ in range(risks)]) if multihead \
                  else create_representation_positive(embed_dim + 1, layers_surv + [risks])
    self.softlog = nn.LogSoftmax(dim = 1)

    self.forward = self.forward_multihead if multihead else self.forward_single

  def forward_multihead(self, x, horizon):
    x_rep = self.embed(x)
    log_beta = self.softlog(self.balance(x_rep)) # Balance

    # Compute cumulative hazard function
    sr = []
    tau_outcome = horizon.clone().detach().requires_grad_(True).unsqueeze(1).to(x.device)
    for outcome_competing in self.outcome:
      outcome = tau_outcome * outcome_competing(torch.cat((x_rep, tau_outcome), 1))
      sr.append(- outcome) # -t * M_r(t, x)

    sr = torch.cat(sr, -1)
    return sr, log_beta, tau_outcome
  
  def gradient(self, outcomes, horizon, e):
    # Compute gradient for points with observed risk - Faster: only one backpropagation
    # the gradient is calculated respect to horizon, as defined in the hazard function
    # outcomes: -t * M_r(t, x), one for each risk
    return grad([- outcomes[:, r][e == (r + 1)].sum() for r in range(self.risks)], horizon, create_graph = True)[0].clamp_(1e-10)[:, 0]

  def forward_single(self, x, horizon):
    x_rep = self.embed(x)
    log_beta = self.softlog(self.balance(x_rep)) # Balance

    # Compute cumulative hazard function
    tau_outcome = horizon.clone().detach().requires_grad_(True).unsqueeze(1)
    outcome = tau_outcome * self.outcome(torch.cat((x_rep, tau_outcome), 1))
    return -outcome, log_beta, tau_outcome
  

class NeuralFineGrayStage2Torch(nn.Module):
  '''
  The model used in 2nd stage of the Neural Fine-Gray model. 
  It only consider the case with multihead = True as chosen in the NeuralFineGray paper.
  Differences from the normal NeuralFineGrayTorch:
    - Need to pass the stage1_model to the __init__() method
        - Only used for inference, so it is set to eval() to avoid training / freeze the parameter learned from stage 1
    - Need to differentiate between features used in the 1st stage (only some features) and the 2nd stage (all features), which requires:
        - Have a parameter called stage1_dim to specify the number of features used in the 1st stage __init__()
        - Make sure the input to the forward() method always has the features used in the 1st stage first to allow for slicing
    - It has its own embedding module, which takes all features as input, and is then added to the embedding from the 1st stage, in forward():
        - x_rep2 = self.embed(x)
        - x_rep1 = self.stage1_model.embed(x[:, :self.stage1_dim])
        - x_rep = x_rep1 + x_rep2
    - Need to have a way to get the contribution from the 1st and 2nd stage separately, which requires access to x_rep1 and x_rep2
  '''

  def __init__(self, stage1_model, stage1_dim,
               inputdim, embed_dim = 32, layers = [64], act = 'ReLU', layers_surv = [64],
               risks = 3, dropout = 0.3, optimizer = "Adam", multihead = True):
    super().__init__()
    self.stage1_model = stage1_model; self.stage1_model.eval() # access embedding module through self.stage1_model.embed
    self.stage1_dim = stage1_dim # number of features used in the 1st stage
    self.input_dim = inputdim
    self.risks = risks  # Competing risks
    self.dropout = dropout
    self.optimizer = optimizer

    self.embed = nn.Sequential(*create_representation(inputdim, layers + [embed_dim], act, self.dropout)) # Assign each point to a cluster
    self.balance = nn.Sequential(*create_representation(embed_dim, layers + [risks], act)) # Define balance between outcome (ensure sum < 1)
    self.outcome = nn.ModuleList(
                      [create_representation_positive(embed_dim + 1, layers_surv + [1]) # Multihead (one for each outcome)
                  for _ in range(risks)]) if multihead \
                  else create_representation_positive(embed_dim + 1, layers_surv + [risks])
    self.softlog = nn.LogSoftmax(dim = 1)

    self.forward = self.forward_multihead # if multihead else self.forward_single

  def embed_by_stage(self, x):
    with torch.no_grad():
      x_rep1 = self.stage1_model.embed(x[:, :self.stage1_dim])
    x_rep2 = self.embed(x)
    return x_rep1, x_rep2

  def forward_multihead(self, x, horizon):
    x_rep1, x_rep2 = self.embed_by_stage(x)
    x_rep = x_rep1 + x_rep2
    log_beta = self.softlog(self.balance(x_rep)) # Balance

    # Compute cumulative hazard function
    sr = []
    tau_outcome = horizon.clone().detach().requires_grad_(True).unsqueeze(1).to(x.device)
    for outcome_competing in self.outcome:
      outcome = tau_outcome * outcome_competing(torch.cat((x_rep, tau_outcome), 1))
      sr.append(- outcome) # -t * M_r(t, x)

    sr = torch.cat(sr, -1)
    return sr, log_beta, tau_outcome
  
  def gradient(self, outcomes, horizon, e):
    # Compute gradient for points with observed risk - Faster: only one backpropagation
    # the gradient is calculated respect to horizon, as defined in the hazard function
    # outcomes: -t * M_r(t, x), one for each risk
    return grad([- outcomes[:, r][e == (r + 1)].sum() for r in range(self.risks)], horizon, create_graph = True)[0].clamp_(1e-10)[:, 0]


class NeuralFineGray(DSMBase):

  def __init__(self, cuda = torch.cuda.is_available(), cause_specific = True, norm_uniform = True, **params):
    self.params = params
    self.fitted = False
    self.cuda = cuda
    self.cause_specific = cause_specific
    self.norm_uniform = norm_uniform
    if cause_specific:
        self.loss = total_loss_cs
    else:
        raise Exception("Only cause_specific = True is supported for NeuralFineGray")

  def _gen_torch_model(self, inputdim, optimizer, risks):
    model = NeuralFineGrayTorch(inputdim, **self.params,
                                     risks = risks,
                                     optimizer = optimizer).double()
    if self.cuda > 0:
      model = model.cuda()
    return model
  
  def _normalise(self, time, save = False):
    time = time.cpu().numpy() if isinstance(time, torch.Tensor) else time
    if self.norm_uniform:
      if save: 
        self.time = np.sort(time)
      ecdf = lambda x: (np.searchsorted(self.time, x, side='right') + 1) / len(self.time)
      uniform_data = torch.Tensor([ecdf(t) for t in time])
      return uniform_data + 1e-5 # Avoid 0
    else:
      time = time + 1 # Do not want event at time 0
      if save: 
        self.max_time = time.max()
      time = torch.Tensor(time)
      return time / self.max_time # Normalise time between 0 and 1

  def fit(self, x, t, e, vsize = 0.15, val_data = None,
          optimizer = "Adam", random_state = 100, **args):
    processed_data = self._preprocess_training_data(x, t, e,
                                                   vsize, val_data,
                                                   random_state)
    x_train, t_train, e_train, x_val, t_val, e_val = processed_data

    t_train = self._normalise(t_train, save = True)
    t_val = self._normalise(t_val)

    maxrisk = int(np.nanmax(e_train.cpu().numpy()))
    self.risks = maxrisk
    model = self._gen_torch_model(x_train.size(1), optimizer, risks = maxrisk)
    model, speed = train_nfg(model, self.loss,
                         x_train, t_train, e_train,
                         x_val, t_val, e_val, cuda = self.cuda,
                         **args)

    self.speed = speed # Number of iterations needed to converge
    self.torch_model = model.eval()
    self.fitted = True
    return self   

  def get_embed_by_stage(self, x):
    x = self._preprocess_test_data(x)
    with torch.no_grad():
      x_rep1, x_rep2 = self.torch_model.embed_by_stage(x)
    return x_rep1, x_rep2

  def compute_nll(self, x, t, e):
    if not self.fitted:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `_eval_nll`.")
    processed_data = self._preprocess_training_data(x, t, e, 0, None, 0)
    _, _, _, x_val, t_val, e_val = processed_data
    t_val = self._normalise(t_val)

    if self.cuda:
      x_val, t_val, e_val = x_val.cuda(), t_val.cuda(), e_val.cuda()

    loss = self.loss(self.torch_model, x_val, t_val, e_val)
    return loss.item()

  def predict_survival(self, x, t, risk = 1):
    x = self._preprocess_test_data(x)
    if not isinstance(t, (list, np.ndarray, torch.Tensor)):
      t = [t]
    if self.fitted:
      scores = []
      for t_ in t:
        t_ = self._normalise(torch.DoubleTensor([t_] * len(x))).to(x.device)
        log_sr, log_beta, _  = self.torch_model(x, t_)
        beta = 1 if self.cause_specific else log_beta.exp() 
        outcomes = 1 - beta * (1 - torch.exp(log_sr)) # Exp diff => Ignore balance but just the risk of one disease
        scores.append(outcomes[:, int(risk) - 1].unsqueeze(1).detach().cpu().numpy())
      return np.concatenate(scores, axis = 1)
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_survival`.")
  
  def compute_gradient(self, x, t, risk = 1):
    x = self._preprocess_test_data(x)
    if not isinstance(t, (list, np.ndarray, torch.Tensor)):
      t = [t]
    if self.fitted:
      gradients = []
      for t_ in t:
        t_ = self._normalise(torch.DoubleTensor([t_] * len(x))).to(x.device)
        log_sr, _, tau = self.torch_model(x, t_)
        log_hr = self.torch_model.gradient(log_sr, tau, torch.DoubleTensor([risk] * len(x)).to(x.device)).log()
        gradients.append(log_hr.unsqueeze(1).detach().cpu().numpy())
      return np.concatenate(gradients, axis = 1)
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `compute_gradient`.")

  def feature_importance(self, x, t, e, n = 100):
    """
      This method computes the features' importance by a  random permutation of the input variables.

      Parameters
      ----------
      x: np.ndarray
          A numpy array of the input features, ( x ).
      t: np.ndarray
          A numpy array of the event/censoring times, ( t ).
      e: np.ndarray
          A numpy array of the event/censoring indicators, ( delta ).
          ( delta = 1 ) means the event took place.
      n: int
          Number of permutations used for the computation

      Returns:
        (dict, dict): Dictionary of the mean impact on likelihood and normal confidence interval

    """
    global_nll = self.compute_nll(x, self._normalise(t), e)
    permutation = np.arange(len(x))
    performances = {j: [] for j in range(x.shape[1])}
    for _ in tqdm(range(n)):
      np.random.shuffle(permutation)
      for j in performances:
        x_permuted = x.copy()
        x_permuted[:, j] = x_permuted[:, j][permutation]
        performances[j].append(self.compute_nll(x_permuted, t, e))
    return {j: np.mean((np.array(performances[j]) - global_nll)/abs(global_nll)) for j in performances}, \
           {j: 1.96 * np.std((np.array(performances[j]) - global_nll)/abs(global_nll)) / np.sqrt(n) for j in performances}
  

class NeuralFineGrayStage2(NeuralFineGray):

  def __init__(self, stage1_model, stage1_dim, 
               cuda = torch.cuda.is_available(), cause_specific = True, norm_uniform = True, **params):
    self.stage1_model = stage1_model
    self.stage1_dim = stage1_dim
    self.params = params
    self.fitted = False
    self.cuda = cuda
    self.cause_specific = cause_specific
    self.norm_uniform = norm_uniform
    if cause_specific:
        self.loss = total_loss_cs
    else:
        raise Exception("Only cause_specific = True is supported for NeuralFineGray")

  def _gen_torch_model(self, stage1_model, stage1_dim, inputdim, optimizer, risks):
    model = NeuralFineGrayStage2Torch(stage1_model, stage1_dim, 
                                      inputdim, **self.params,
                                     risks = risks,
                                     optimizer = optimizer).double()
    if self.cuda > 0:
      model = model.cuda()
    return model

  def fit(self, x, t, e, vsize = 0.15, val_data = None,
          optimizer = "Adam", random_state = 100, **args):
    processed_data = self._preprocess_training_data(x, t, e,
                                                   vsize, val_data,
                                                   random_state)
    x_train, t_train, e_train, x_val, t_val, e_val = processed_data

    t_train = self._normalise(t_train, save = True)
    t_val = self._normalise(t_val)

    maxrisk = int(np.nanmax(e_train.cpu().numpy()))
    self.risks = maxrisk
    model = self._gen_torch_model(self.stage1_model, self.stage1_dim, x_train.size(1), optimizer, risks = maxrisk)
    model, speed = train_nfg(model, self.loss,
                         x_train, t_train, e_train,
                         x_val, t_val, e_val, cuda = self.cuda,
                         **args)

    self.speed = speed # Number of iterations needed to converge
    self.torch_model = model.eval()
    self.fitted = True
    return self    

  def compute_nll(self, x, t, e):
    if not self.fitted:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `_eval_nll`.")
    processed_data = self._preprocess_training_data(x, t, e, 0, None, 0)
    _, _, _, x_val, t_val, e_val = processed_data
    t_val = self._normalise(t_val)

    if self.cuda:
      x_val, t_val, e_val = x_val.cuda(), t_val.cuda(), e_val.cuda()

    loss = self.loss(self.torch_model, x_val, t_val, e_val)
    return loss.item()

  def predict_survival(self, x, t, risk = 1):
    x = self._preprocess_test_data(x)
    if not isinstance(t, (list, np.ndarray, torch.Tensor)):
      t = [t]
    if self.fitted:
      scores = []
      for t_ in t:
        t_ = self._normalise(torch.DoubleTensor([t_] * len(x))).to(x.device)
        log_sr, log_beta, _  = self.torch_model(x, t_)
        beta = 1 if self.cause_specific else log_beta.exp() 
        outcomes = 1 - beta * (1 - torch.exp(log_sr)) # Exp diff => Ignore balance but just the risk of one disease
        scores.append(outcomes[:, int(risk) - 1].unsqueeze(1).detach().cpu().numpy())
      return np.concatenate(scores, axis = 1)
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_survival`.")
