from SurvivalEVAL import LifelinesEvaluator
import pandas as pd
import numpy as np
from scipy.integrate import trapezoid
import warnings
import torch
import matplotlib.pyplot as plt
from abc import ABC
from SurvivalEVAL.Evaluations.custom_types import NumericArrayLike
from typing import Union, Optional, Callable

# Section 4.1  Performance metrics in "Random survival forests for competing risks"
# Hemant Ishwaran, Thomas A. Gerds, Udaya B. Kogalur, Richard D. Moore, Stephen J. Gange, Bryan M. Lau, 
def c_index_cr(event_times, predicted_scores, event_observed, event_of_interest=1):
    '''
    Compute the concordance index for competing risks.
    Args:
        event_times: (n,) array of event times
        predicted_scores: (n, k) array of predicted scores
        event_observed: (n,) array of event observed indicator
        event_of_interest: int, the event of interest
    '''
    num_correct, num_tied, num_pairs = 0, 0, 0
    for i in range(len(event_times)):
        if event_observed[i] != event_of_interest:
            continue
        for j in range(len(event_times)):
            # according to observation: i should have higher risk than j
            if event_times[i] < event_times[j] or (event_observed[j] > 0 and event_observed[j] != event_of_interest):
                num_pairs += 1
                if predicted_scores[i] > predicted_scores[j]:
                    num_correct += 1
            else:
                num_tied += 1
    # print(num_correct, num_tied, num_pairs)
    if num_pairs == 0:
        raise ZeroDivisionError("No admissable pairs in the dataset.")
    return (num_correct + num_tied / 2) / num_pairs

class EvaluatorWithMaxTime(LifelinesEvaluator, ABC):
    def __init__(self,
            surv: pd.DataFrame,
            test_event_times: NumericArrayLike,
            test_event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None,
            predict_time_method: str = "Median",
            interpolation: str = "Linear",
            max_time: Optional[float] = None):
        super(EvaluatorWithMaxTime, self).__init__(surv, test_event_times, test_event_indicators, train_event_times,
                                                 train_event_indicators, predict_time_method, interpolation)
        self.max_time = max_time

    def integrated_brier_score(
            self,
            num_points: int = None,
            IPCW_weighted: bool = True,
            draw_figure: bool = False
    ) -> float:
        """
        Calculate the integrated Brier score (IBS) from the predicted survival curve.
        param num_points: int, default = None
            Number of points at which the Brier score is to be calculated. If None, the number of points is set to
            the number of event/censor times from the training and test sets.
        param IPCW_weighted: bool, default = True
            Whether to use IPCW weighting for the Brier score.
        param draw_figure: bool, default = False
            Whether to draw the figure of the IBS.
        :return: float
            The integrated Brier score.
        """
        if IPCW_weighted:
            self._error_trainset("IPCW-weighted Integrated Brier Score (IBS)")

        max_target_time = np.max(np.concatenate((self.event_times, self.train_event_times))) if self.train_event_times \
            is not None else np.max(self.event_times)

        # If number of target time is not indicated, then we use the censored times obtained from test set
        if num_points is None:
            censored_times = self.event_times[self.event_indicators == 0]
            if self.max_time is not None:
                censored_times = censored_times[censored_times <= self.max_time]
            time_points = np.unique(censored_times)
            if time_points.size == 0:
                raise ValueError("You don't have censor data in the testset, "
                                 "please provide \"num_points\" for calculating IBS")
            else:
                time_range = np.max(time_points) - np.min(time_points)
        else:
            time_points = np.linspace(0, max_target_time, num_points)
            time_range = max_target_time

        # Get single brier score from multiple target times, and use trapezoidal integral to calculate ISB.
        b_scores = self.brier_score_multiple_points(time_points, IPCW_weighted)
        if np.isnan(b_scores).any():
            warnings.warn("Time-dependent Brier Score contains nan")
            bs_dict = {}
            for time_point, b_score in zip(time_points, b_scores):
                bs_dict[time_point] = b_score
            print("Brier scores for multiple time points are".format(bs_dict))
        integral_value = trapezoid(b_scores, time_points)
        ibs_score = integral_value / time_range

        # Draw the Brier score graph
        if draw_figure:
            plt.plot(time_points, b_scores, 'bo-')
            score_text = r'IBS$= {:.3f}$'.format(ibs_score)
            plt.plot([], [], ' ', label=score_text)
            plt.legend()
            # plt.text(500, 0.05, r'IBS$= {:.3f}$'.format(ibs_score), verticalalignment='top',
            #          horizontalalignment='left', fontsize=12, color='Black')
            plt.xlabel('Time')
            plt.ylabel('Brier Score')
            plt.show()
        return ibs_score


def constant_density_interpolate(s, sub=10):
    '''
    Courtesy of https://github.com/havakv/pycox/blob/master/pycox/models/interpolation.py#L68
    s: torch.tensor, shape = (n, m) or (m,)
    sub: int, number of sub-intervals to interpolate; e.g., if sub=10, then 10 sub-intervals are interpolated between each pair of adjacent points
    '''
    # if s in numpy array, convert to torch tensor
    if isinstance(s, np.ndarray):
        s = torch.tensor(s)
    if s.dim() == 1:
        s = s.unsqueeze(0)
    n, m = s.shape
    diff = (s[:, 1:] - s[:, :-1]).contiguous().view(-1, 1).repeat(1, sub).view(n, -1)
    rho = torch.linspace(0, 1, sub+1)[:-1].contiguous().repeat(n, m-1)
    s_prev = s[:, :-1].contiguous().view(-1, 1).repeat(1, sub).view(n, -1)
    s_interpolated = torch.zeros(n, int((m-1)*sub + 1))
    s_interpolated[:, :-1] = diff * rho + s_prev
    s_interpolated[:, -1] = s[:, -1]
    return s_interpolated.numpy()


def sort_by_time(surv_pred_event, temp_test_time, temp_test_event):
    '''
    Sort by time to make c index calculate faster @Shi-ang may add into the Evaluator
    surv_pred_event: Dataframe
    temp_test_time: np.array
    temp_test_event: np.array
    '''
    surv_pred_event['time'] = temp_test_time
    surv_pred_event['event'] = temp_test_event
    surv_pred_event = surv_pred_event.sort_values('time')

    temp_test_time = surv_pred_event['time'].to_numpy()
    temp_test_event = surv_pred_event['event'].to_numpy()
    surv_pred_event = surv_pred_event.drop(['time', 'event'], axis=1)
    return surv_pred_event, temp_test_time, temp_test_event

def all_events_ci(mod_out, test_time, test_event):
    '''
    all events
    mod_out: List of surv pred
    test_time: np.array of float/int #patient, #event
    test_event: np.array of binary #patient, #event
    '''
    surv_pred_event = pd.concat([pd.DataFrame(surv_pred) for surv_pred in mod_out])
    surv_pred_event = surv_pred_event.reset_index(drop=True)
    temp_test_time = np.concatenate([test_time[:, event_id] for event_id in range(test_time.shape[1])])
    temp_test_event = np.concatenate([test_event[:, event_id] for event_id in range(test_event.shape[1])])

    surv_pred_event, temp_test_time, temp_test_event = sort_by_time(surv_pred_event, temp_test_time, temp_test_event)
    evaluator = LifelinesEvaluator(surv_pred_event.T, temp_test_time, temp_test_event)

    cindex, _, _ = evaluator.concordance()
    return cindex

def global_C_index(mod_out, test_time, test_event, weight=True):
    '''
    each events
    mod_out: List of surv pred
    test_time: np.array of float/int #patient, #event
    test_event: np.array of binary #patient, #event
    '''
    cindex_list = []
    global_total_pairs = 0.0
    global_concordant_pairs = 0.0
    for event_id in range(len(mod_out)):
        surv_pred_event = pd.DataFrame(mod_out[event_id])
        temp_test_time = test_time[:,event_id]
        temp_test_event = test_event[:,event_id]

        surv_pred_event, temp_test_time, temp_test_event = sort_by_time(surv_pred_event, temp_test_time, temp_test_event)
        evaluator = LifelinesEvaluator(surv_pred_event.T, temp_test_time, temp_test_event)

        cindex, concordant_pairs, total_pairs = evaluator.concordance()
        cindex_list.append(cindex)
        global_total_pairs += total_pairs
        global_concordant_pairs += concordant_pairs
    if weight:
        return global_concordant_pairs/global_total_pairs
    else:
        return np.mean(cindex_list)

def local_C_index(mod_out, test_time, test_event, weight=True):
    '''
    each patient
    mod_out: List of surv pred
    test_time: np.array of float/int #patient, #event
    test_event: np.array of binary #patient, #event
    '''
    cindex_list = []
    global_total_pairs = 0.0
    global_concordant_pairs = 0.0
    for patient_id in range(test_time.shape[0]):
        surv_pred_patient = np.column_stack([mod_out[event_index][patient_id, :] for event_index in range(len(mod_out))]).T

        surv_pred_event = pd.DataFrame(surv_pred_patient)
        temp_test_time = test_time[patient_id,:]
        temp_test_event = test_event[patient_id,:]
        if np.sum(temp_test_event) != 0:
            try:
                evaluator = LifelinesEvaluator(surv_pred_event.T, temp_test_time, temp_test_event)
                cindex, concordant_pairs, total_pairs = evaluator.concordance()
                cindex_list.append(cindex)
                global_total_pairs += total_pairs
                global_concordant_pairs += concordant_pairs
            except:
                continue    
    if weight:
        return global_concordant_pairs/global_total_pairs
    else:
        return np.mean(cindex_list)
