import numpy as np
import torch

from .combiner import Combiner
from .creation import combiners


class Energy(Combiner):
    def __init__(self, base_model, new_model, dataset, **kwargs):
        super().__init__(base_model, new_model, dataset)

        self._setup()

    def predict(self, dataloader):
        all_base_energies, all_new_energies, all_base_preds, all_new_preds = self._get_energies(dataloader)

        meta_preds = []

        for i in range(len(all_new_preds)):
            if all_base_energies[i] < all_new_energies[i]:
                meta_preds.append(all_base_preds[i])
            else:
                meta_preds.append(all_new_preds[i])

        meta_preds = torch.tensor(meta_preds).cpu().numpy()

        return meta_preds

    def get_choices(self, dataloader):
        all_base_energies, all_new_energies, all_base_preds, all_new_preds = self._get_energies(dataloader)

        meta_preds = []
        meta_choices = []

        for i in range(len(all_new_preds)):
            if all_base_energies[i] < all_new_energies[i]:
                meta_preds.append(all_base_preds[i])
                meta_choices.append(0)
            else:
                meta_preds.append(all_new_preds[i])
                meta_choices.append(1)

        meta_preds = np.array(meta_preds)
        meta_choices = np.array(meta_choices)

        return meta_preds, all_base_preds, all_new_preds, meta_choices

    def _get_energies(self, dataloader):
        all_base_energies = []
        all_new_energies = []

        all_base_preds = []
        all_new_preds = []

        with torch.no_grad():
            for x, _, _, _ in dataloader:
                base_logits = self.base_model(x)
                new_logits = self.new_model(x)

                base_energy = energy_score(base_logits)
                new_energy = energy_score(new_logits)
                all_base_energies.append(base_energy)
                all_new_energies.append(new_energy)

                all_base_preds.append(torch.argmax(base_logits, dim=1))
                all_new_preds.append(torch.argmax(new_logits, dim=1))

        all_base_energies = torch.cat(all_base_energies).cpu().numpy()
        all_new_energies = torch.cat(all_new_energies).cpu().numpy()

        all_base_preds = torch.cat(all_base_preds).cpu().numpy()
        all_new_preds = torch.cat(all_new_preds).cpu().numpy()

        return all_base_energies, all_new_energies, all_base_preds, all_new_preds

    def _setup(self):
        pass


def energy_score(logits):
    return - torch.log(torch.sum(torch.exp(logits), dim=1))


combiners.register_builder("energy", Energy)
