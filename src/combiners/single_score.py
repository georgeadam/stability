import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from .combiner import Combiner
from .creation import combiners
from .scorers import scorers


class SingleScore(Combiner):
    def __init__(self, base_model, new_model, dataset, base_prediction_tracker, new_prediction_tracker, scorer_args,
                 **kwargs):
        super().__init__(base_model, new_model, dataset)
        self.scorer_args = scorer_args
        self.base_predictions = base_prediction_tracker.get_validation_predictions()
        self.new_predictions = new_prediction_tracker.get_validation_predictions()

        self.base_nearest_neighbors = None
        self.new_nearest_neighbors = None

        self.base_scores = None
        self.new_scores = None

        self._setup()

    def predict(self, dataloader):
        with torch.no_grad():
            base_preds, new_preds = self._get_preds(dataloader)
            base_scores, new_scores = self._get_scores(dataloader)

        stacked_preds = np.stack([base_preds, new_preds], axis=1)
        sample_indices = np.arange(len(stacked_preds))

        meta_choices = (new_scores > base_scores).astype(int)
        meta_preds = stacked_preds[sample_indices, meta_choices]

        return meta_preds

    def get_choices(self, dataloader, base_temperature=1.0, new_temperature=1.0):
        with torch.no_grad():
            base_preds, new_preds = self._get_preds(dataloader, base_temperature, new_temperature)
            base_scores, new_scores = self._get_scores(dataloader)

        stacked_preds = np.stack([base_preds, new_preds], axis=1)
        sample_indices = np.arange(len(stacked_preds))

        meta_choices = (new_scores > base_scores).int()
        meta_preds = stacked_preds[sample_indices, meta_choices]

        return meta_preds, base_preds, new_preds, meta_choices

    def _get_scores(self, dataloader):
        base_embeddings = extract_embeddings(dataloader, self.base_model)
        base_neighbor_indices = self.base_nearest_neighbors.kneighbors(base_embeddings)[1]
        base_scores = self.base_scores[base_neighbor_indices]
        base_scores = np.mean(base_scores, axis=1)

        new_embeddings = extract_embeddings(dataloader, self.new_model)
        new_neighbor_indices = self.new_nearest_neighbors.kneighbors(new_embeddings)[1]
        new_scores = self.new_scores[new_neighbor_indices]
        new_scores = np.mean(new_scores, axis=1)

        return base_scores, new_scores

    def _get_preds(self, dataloader, base_temperature=1.0, new_temperature=1.0):
        all_base_preds = []
        all_new_preds = []

        for x, _, _, _ in dataloader:
            base_out = self.base_model(x)
            base_probs = torch.nn.Softmax(dim=1)(base_out / base_temperature)
            new_out = self.new_model(x)
            new_probs = torch.nn.Softmax(dim=1)(new_out / new_temperature)

            all_base_preds.append(torch.argmax(base_probs, dim=1))
            all_new_preds.append(torch.argmax(new_probs, dim=1))

        all_base_preds = torch.cat(all_base_preds).cpu().numpy()
        all_new_preds = torch.cat(all_new_preds).cpu().numpy()

        return all_base_preds, all_new_preds

    def _setup(self):
        scorer = scorers.create(self.scorer_args.name, **self.scorer_args.params)

        self.base_scores = scorer.generate_scores(self.base_predictions)
        self.new_scores = scorer.generate_scores(self.new_predictions)

        base_embeddings = extract_embeddings(self.dataset.val_dataloader(), self.base_model)
        base_nearest_neighbors = NearestNeighbors(n_neighbors=10)
        self.base_nearest_neighbors = base_nearest_neighbors.fit(base_embeddings)

        new_embeddings = extract_embeddings(self.dataset.val_dataloader(), self.new_model)
        new_nearest_neighbors = NearestNeighbors(n_neighbors=10)
        self.new_nearest_neighbors = new_nearest_neighbors.fit(new_embeddings)


@torch.no_grad()
def extract_embeddings(dataloader, model):
    embeddings = []

    for x, _, _, _ in dataloader:
        embedding = model.forward_embedding(x)
        embeddings.append(embedding)

    return torch.cat(embeddings).detach().cpu().numpy()


combiners.register_builder("single_score", SingleScore)
