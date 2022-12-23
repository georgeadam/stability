import numpy as np

from .creation import scorers
from .scorer import ScorerInterface


class SAvg(ScorerInterface):
    def generate_scores(self, predictions):
        predictions = predictions["preds"]
        scores = [compute_s_avg(pred) for pred in predictions]

        return np.array(scores)


def compute_s_avg(labels, k=1.0):
    a_t = []
    L = labels[-1]

    for l in labels:
        if l == L:
            a_t.append(0)
        else:
            a_t.append(1)

    a_t = np.array(a_t, dtype='f')

    v_t = np.ones(len(labels)) - np.linspace(0, 1, len(labels)) ** k

    if np.sum(a_t) == 0.0:
        s_avg = v_t[0]

    else:
        s_avg = (1 / np.sum(a_t)) * np.sum(a_t * v_t)

    return s_avg


scorers.register_builder("savg", SAvg)
