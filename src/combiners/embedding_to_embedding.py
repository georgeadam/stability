import abc

import numpy as np
import torch

from .mappers import mappers
from .creation import combiners


class EmbeddingToEmbedding(metaclass=abc.ABCMeta):
    def __init__(self, base_model, new_model, dataset, mapper_args):
        self.base_model = base_model
        self.new_model = new_model
        self.dataset = dataset
        self.mapper = None

        self._setup(mapper_args)

    def predict(self, dataloader):
        with torch.no_grad():
            base_probs, new_probs = self._get_probs(dataloader)

        base_confidences = probs_to_confidence(base_probs)
        new_confidences = probs_to_confidence(new_probs)

        base_preds = np.argmax(base_probs, axis=1)
        new_preds = np.argmax(new_probs, axis=1)
        combined_preds = np.stack([base_preds, new_preds], axis=1)

        meta_pred_indices = (new_confidences > base_confidences).astype(int)
        batch_indices = np.arange(len(combined_preds))
        meta_preds = combined_preds[batch_indices, meta_pred_indices]

        return meta_preds

    def predict_frankenstein(self, dataloader):
        with torch.no_grad():
            base_probs, new_probs = self._get_probs(dataloader)

        frankenstein_preds = np.argmax(base_probs, axis=1)

        return frankenstein_preds

    def _get_probs(self, dataloader):
        all_base_probs = []
        all_new_probs = []

        self.base_model = self.base_model.to("cuda:0")
        self.new_model = self.new_model.to("cuda:0")

        for x, _, _, _ in dataloader:
            x = x.to("cuda:0")
            new_out = self.new_model(x)
            new_probs = torch.nn.Softmax(dim=1)(new_out)
            new_embeddings = self.new_model.forward_embedding(x)

            base_embeddings = self.mapper(new_embeddings.cpu())
            base_out = self.base_model.forward_classifier(base_embeddings.to(x.device))
            base_probs = torch.nn.Softmax(dim=1)(base_out)

            all_base_probs.append(base_probs)
            all_new_probs.append(new_probs)

        all_base_probs = torch.cat(all_base_probs).cpu().numpy()
        all_new_probs = torch.cat(all_new_probs).cpu().numpy()

        self.base_model = self.base_model.to("cpu")
        self.new_model = self.new_model.to("cpu")

        return all_base_probs, all_new_probs

    def _extract_embeddings(self, model, dataloader):
        all_embeddings = []
        model = model.to("cuda:0")

        with torch.no_grad():
            for x, _, _, _ in dataloader:
                x = x.to("cuda:0")
                embeddings = model.forward_embedding(x)
                all_embeddings.append(embeddings)

        all_embeddings = torch.cat(all_embeddings).cpu()
        model = model.to("cpu")

        return all_embeddings

    def _setup(self, mapper_args):
        print("Extracting Embeddings")
        new_embeddings = self._extract_embeddings(self.new_model, self.dataset.val_dataloader())
        base_embeddings = self._extract_embeddings(self.base_model, self.dataset.val_dataloader())
        print("Finished Extracting Embeddings")

        print("Fitting Mapper")
        self.mapper = mappers.create(mapper_args.name, **mapper_args.params)
        self.mapper.fit(new_embeddings, base_embeddings)
        print("Finished Fitting Mapper")


def probs_to_confidence(probs):
    return np.max(probs, axis=1)


combiners.register_builder("embedding_to_embedding", EmbeddingToEmbedding)
