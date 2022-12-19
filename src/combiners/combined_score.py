import numpy as np
import torch

from .combiner import Combiner
from .creation import combiners

from sklearn.neighbors import NearestNeighbors


class Combined(Combiner):
    def __init__(self, base_model, new_model, dataset, base_prediction_tracker, new_prediction_tracker, **kwargs):
        super().__init__(base_model, new_model, dataset)
        self.base_predictions = base_prediction_tracker.validation_predictions
        self.base_predictions = self.base_predictions["preds"].to_numpy().reshape((-1, int(self.base_predictions["epoch"].max() + 1)), order="F")

        self.new_predictions = new_prediction_tracker.validation_predictions
        self.new_predictions = self.new_predictions["preds"].to_numpy().reshape((-1, int(self.new_predictions["epoch"].max() + 1)), order="F")

        self.base_nearest_neighbors = None
        self.new_nearest_neighbors = None

        self.base_scores = None
        self.new_scores = None

        self._setup()

    def predict(self, dataloader):
        with torch.no_grad():
            base_confidences, new_confidences, base_preds, new_preds = self._get_confidences(dataloader)
            base_scores, new_scores = self._get_savg_scores(dataloader)

        stacked_preds = np.stack([base_preds, new_preds], axis=1)
        sample_indices = np.arange(len(stacked_preds))

        meta_choices = ((new_confidences > base_confidences) & (new_scores > base_scores)).astype(int)
        meta_preds = stacked_preds[sample_indices, meta_choices]

        return meta_preds

    def get_choices(self, dataloader, base_temperature=1.0, new_temperature=1.0):
        with torch.no_grad():
            base_confidences, new_confidences, base_preds, new_preds = self._get_confidences(dataloader,
                                                                                             base_temperature,
                                                                                             new_temperature)
            base_scores, new_scores = self._get_savg_scores(dataloader)

        stacked_preds = np.stack([base_preds, new_preds], axis=1)
        sample_indices = np.arange(len(stacked_preds))

        meta_choices = ((new_confidences > base_confidences) & (new_scores > base_scores)).int()
        meta_preds = stacked_preds[sample_indices, meta_choices]

        return meta_preds, base_preds, new_preds, meta_choices

    def _get_savg_scores(self, dataloader):
        base_embeddings = extract_embeddings(dataloader, self.base_model)
        base_neighbor_indices = self.base_nearest_neighbors.kneighbors(base_embeddings)[1]
        base_scores = self.base_scores[base_neighbor_indices]
        base_scores = np.mean(base_scores, axis=1)

        new_embeddings = extract_embeddings(dataloader, self.new_model)
        new_neighbor_indices = self.new_nearest_neighbors.kneighbors(new_embeddings)[1]
        new_scores = self.new_scores[new_neighbor_indices]
        new_scores = np.mean(new_scores, axis=1)

        return base_scores, new_scores

    def _get_confidences(self, dataloader, base_temperature=1.0, new_temperature=1.0):
        all_base_confidences = []
        all_new_confidences = []

        all_base_preds = []
        all_new_preds = []

        for x, _, _, _ in dataloader:
            base_out = self.base_model(x)
            base_probs = torch.nn.Softmax(dim=1)(base_out / base_temperature)
            new_out = self.new_model(x)
            new_probs = torch.nn.Softmax(dim=1)(new_out / new_temperature)

            base_confidence = confidence(base_probs)
            new_confidence = confidence(new_probs)
            all_base_confidences.append(base_confidence)
            all_new_confidences.append(new_confidence)

            all_base_preds.append(torch.argmax(base_probs, dim=1))
            all_new_preds.append(torch.argmax(new_probs, dim=1))

        all_base_confidences = torch.cat(all_base_confidences).cpu().numpy()
        all_new_confidences = torch.cat(all_new_confidences).cpu().numpy()

        all_base_preds = torch.cat(all_base_preds).cpu().numpy()
        all_new_preds = torch.cat(all_new_preds).cpu().numpy()

        return all_base_confidences, all_new_confidences, all_base_preds, all_new_preds

    def _setup(self):
        base_scores = [compute_s_avg(pred) for pred in self.base_predictions]
        self.base_scores = np.array(base_scores)

        new_scores = [compute_s_avg(pred) for pred in self.new_predictions]
        self.new_scores = np.array(new_scores)

        base_embeddings = extract_embeddings(self.dataset.val_dataloader(), self.base_model)
        base_nearest_neighbors = NearestNeighbors(n_neighbors=10)
        self.base_nearest_neighbors = base_nearest_neighbors.fit(base_embeddings)

        new_embeddings = extract_embeddings(self.dataset.val_dataloader(), self.new_model)
        new_nearest_neighbors = NearestNeighbors(n_neighbors=10)
        self.new_nearest_neighbors = new_nearest_neighbors.fit(new_embeddings)


def confidence(probs):
    return torch.max(probs, dim=1).values


def compute_s_avg(labels, k=0.001):
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


@torch.no_grad()
def extract_embeddings(dataloader, model):
    embeddings = []

    for x, _, _, _ in dataloader:
        embedding = model.forward_embedding(x)
        embeddings.append(embedding)

    return torch.cat(embeddings).detach().cpu().numpy()


combiners.register_builder("combined", Combined)
