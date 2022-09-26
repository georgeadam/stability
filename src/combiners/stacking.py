import abc

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .creation import combiners


class Stacking(metaclass=abc.ABCMeta):
    def __init__(self, base_model, new_model, dataset):
        self.base_model = base_model
        self.new_model = new_model
        self.dataset = dataset

        self.normalizer = None
        self.stacker = None

        self._setup()

    def predict(self, dataloader):
        all_base_logits, all_new_logits, combined_logits = self._get_logits(dataloader)

        meta_preds = self.stacker.predict(self.normalizer.transform(combined_logits))

        return meta_preds

    def get_choices(self, dataloader):
        all_base_logits, all_new_logits, combined_logits = self._get_logits(dataloader)

        meta_preds = self.stacker.predict(self.normalizer.transform(combined_logits))
        base_preds = torch.argmax(all_base_logits, dim=1).cpu().numpy()
        new_preds = torch.argmax(all_new_logits, dim=1).cpu().numpy()

        meta_choice = []

        for i in range(len(meta_preds)):
            if meta_preds[i] == base_preds[i]:
                meta_choice.append(0)
            elif meta_preds[i] == new_preds[i]:
                meta_choice.append(1)
            else:
                meta_choice.append(2)

        meta_choice = np.array(meta_choice)

        return meta_preds, base_preds, new_preds, meta_choice

    def _get_logits(self, dataloader):
        all_base_logits = []
        all_new_logits = []

        with torch.no_grad():
            for x, _, _, _ in dataloader:
                base_logits = self.base_model(x)
                new_logits = self.new_model(x)

                all_base_logits.append(base_logits)
                all_new_logits.append(new_logits)

        all_base_logits = torch.cat(all_base_logits)
        all_new_logits = torch.cat(all_new_logits)
        combined_logits = torch.cat([all_base_logits, all_new_logits], dim=1)

        return all_base_logits, all_new_logits, combined_logits

    def _setup(self):
        self.normalizer = StandardScaler()
        self.stacker = LogisticRegression(solver="newton-cg", max_iter=1000000)

        all_base_logits = []
        all_new_logits = []
        all_y = []

        with torch.no_grad():
            for x, y, _, _ in self.dataset.val_dataloader():
                base_logits = self.base_model(x)
                new_logits = self.new_model(x)

                all_base_logits.append(base_logits)
                all_new_logits.append(new_logits)
                all_y.append(y)

        all_base_logits = torch.cat(all_base_logits)
        all_new_logits = torch.cat(all_new_logits)
        all_y = torch.cat(all_y)
        combined_logits = torch.cat([all_base_logits, all_new_logits], dim=1)

        self.normalizer.fit(combined_logits)
        self.stacker.fit(self.normalizer.transform(combined_logits), all_y)


combiners.register_builder("stacking", Stacking)
