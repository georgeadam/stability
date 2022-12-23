import numpy as np

from .creation import scorers
from .scorer import ScorerInterface


class AverageConfidence(ScorerInterface):
    def generate_scores(self, predictions):
        final_epoch_predictions = predictions["preds"][:, -1]
        sample_indices = np.arange(len(final_epoch_predictions))
        probabilities = predictions["probs_full"][sample_indices, :, final_epoch_predictions]

        return np.mean(probabilities, axis=1)


scorers.register_builder("average_confidence", AverageConfidence)
