import abc

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .creation import combiners


class EarlyStacking(metaclass=abc.ABCMeta):
    def __init__(self, base_model, new_model, dataset, preference, model):
        self.base_model = base_model
        self.new_model = new_model
        self.dataset = dataset
        self.preference = preference
        self.model = model

        self.normalizer = None
        self.stacker = None

        self._setup()

    def predict(self, dataloader):
        all_base_preds = []
        all_new_preds = []
        all_x = []

        with torch.no_grad():
            for x, _, _, _ in dataloader:
                base_logits = self.base_model(x)
                base_preds = torch.argmax(base_logits, dim=1)
                new_logits = self.new_model(x)
                new_preds = torch.argmax(new_logits, dim=1)

                all_base_preds.append(base_preds)
                all_new_preds.append(new_preds)
                all_x.append(x.view(x.shape[0], -1))

        all_base_preds = torch.cat(all_base_preds)
        all_new_preds = torch.cat(all_new_preds)
        combined_preds = torch.stack([all_base_preds, all_new_preds], dim=1)
        all_x = torch.cat(all_x)

        model_choice = self.stacker.predict(self.normalizer.transform(all_x))
        sample_indices = torch.arange(len(combined_preds))
        meta_preds = combined_preds[sample_indices, model_choice]

        return meta_preds.cpu().numpy()

    def _setup(self):
        self.normalizer = StandardScaler()

        if self.model == "lr":
            self.stacker = LogisticRegression(solver="newton-cg", max_iter=1000000)
        elif self.model == "svm":
            self.stacker = SVC()

        all_base_correct = []
        all_new_correct = []
        all_x = []

        with torch.no_grad():
            for x, y, _, _ in self.dataset.val_dataloader():
                base_logits = self.base_model(x)
                base_preds = torch.argmax(base_logits, dim=1)
                base_correct = base_preds == y

                new_logits = self.new_model(x)
                new_preds = torch.argmax(new_logits, dim=1)
                new_correct = new_preds == y

                all_base_correct.append(base_correct)
                all_new_correct.append(new_correct)
                all_x.append(x.view(x.shape[0], -1))

        all_base_correct = torch.cat(all_base_correct)
        all_new_correct = torch.cat(all_new_correct)
        all_x = torch.cat(all_x)

        stacking_targets = self._create_stacking_targets(all_base_correct, all_new_correct)

        self.normalizer.fit(all_x)
        self.stacker.fit(self.normalizer.transform(all_x), stacking_targets)

    def _create_stacking_targets(self, base_correct, new_correct):
        if self.preference == "base":
            source_indices = []

            for i in range(len(base_correct)):
                if (base_correct[i] and new_correct[i]) or base_correct[i]:
                    source_indices.append(0)
                elif new_correct[i]:
                    source_indices.append(1)
                else:
                    source_indices.append(0)
        elif self.preference == "new":
            source_indices = []

            for i in range(len(base_correct)):
                if (base_correct[i] and new_correct[i]) or new_correct[i]:
                    source_indices.append(1)
                elif base_correct[i]:
                    source_indices.append(0)
                else:
                    source_indices.append(1)

            source_indices = torch.tensor(source_indices)

        return torch.tensor(source_indices)


combiners.register_builder("early_stacking", EarlyStacking)
