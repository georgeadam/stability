import abc

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

        meta_preds = self.stacker.predict(self.normalizer.transform(combined_logits))

        return meta_preds

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
