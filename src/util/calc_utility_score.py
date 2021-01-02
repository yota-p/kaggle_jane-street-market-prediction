# https://www.kaggle.com/gogo827jz/jane-street-super-fast-utility-score-function/data
import numpy as np
from numba import njit


@njit(fastmath=True)
def utility_score_numba(date, weight, resp, action):
    Pi = np.bincount(date, weight * resp * action)
    t = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / len(Pi))
    u = min(max(t, 0), 6) * np.sum(Pi)
    return u
