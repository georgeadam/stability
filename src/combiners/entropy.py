import numpy as np
import torch

from .combiner import Combiner
from .creation import combiners


class Entropy(Combiner):
    def __init__(self, base_model, new_model, dataset):
        super().__init__(base_model, new_model, dataset)

        self._setup()

    def predict(self, dataloader):
        all_base_entropies, all_new_entropies, all_base_preds, all_new_preds = self._get_entropies(dataloader)

        meta_preds = []

        for i in range(len(all_new_preds)):
            if all_base_entropies[i] < all_new_entropies[i]:
                meta_preds.append(all_base_preds[i])
            else:
                meta_preds.append(all_new_preds[i])

        meta_preds = torch.tensor(meta_preds).cpu().numpy()

        return meta_preds

    def get_choices(self, dataloader):
        all_base_entropies, all_new_entropies, all_base_preds, all_new_preds = self._get_entropies(dataloader)

        meta_preds = []
        meta_choices = []

        for i in range(len(all_new_preds)):
            if all_base_entropies[i] < all_new_entropies[i]:
                meta_preds.append(all_base_preds[i])
                meta_choices.append(0)
            else:
                meta_preds.append(all_new_preds[i])
                meta_choices.append(1)

        meta_preds = np.array(meta_preds)
        meta_choices = np.array(meta_choices)

        return meta_preds, all_base_preds, all_new_preds, meta_choices

    def _get_entropies(self, dataloader):
        all_base_entropies = []
        all_new_entropies = []

        all_base_preds = []
        all_new_preds = []

        with torch.no_grad():
            for x, _, _, _ in dataloader:
                base_out = self.base_model(x)
                base_probs = torch.nn.Softmax(dim=1)(base_out)
                new_out = self.new_model(x)
                new_probs = torch.nn.Softmax(dim=1)(new_out)

                base_entropy = entropy(base_probs)
                new_entropy = entropy(new_probs)
                all_base_entropies.append(base_entropy)
                all_new_entropies.append(new_entropy)

                all_base_preds.append(torch.argmax(base_probs, dim=1))
                all_new_preds.append(torch.argmax(new_probs, dim=1))

        all_base_entropies = torch.cat(all_base_entropies).cpu().numpy()
        all_new_entropies = torch.cat(all_new_entropies).cpu().numpy()

        all_base_preds = torch.cat(all_base_preds).cpu().numpy()
        all_new_preds = torch.cat(all_new_preds).cpu().numpy()

        return all_base_entropies, all_new_entropies, all_base_preds, all_new_preds

    def _setup(self):
        pass


def entropy(x):
    return - torch.sum(x * torch.log(x), dim=1)


combiners.register_builder("entropy", Entropy)
