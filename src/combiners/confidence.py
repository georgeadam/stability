import numpy as np
import torch

from .combiner import Combiner
from .creation import combiners


class Confidence(Combiner):
    def __init__(self, base_model, new_model, dataset):
        super().__init__(base_model, new_model, dataset)

        self._setup()

    def predict(self, dataloader):
        with torch.no_grad():
            all_base_confidences, all_new_confidences, all_base_preds, all_new_preds = self._get_confidences(dataloader)

        meta_preds = []

        for i in range(len(all_new_preds)):
            if all_base_confidences[i] > all_new_confidences[i]:
                meta_preds.append(all_base_preds[i])
            else:
                meta_preds.append(all_new_preds[i])

        meta_preds = torch.tensor(meta_preds).cpu().numpy()

        return meta_preds

    def get_choices(self, dataloader, base_temperature=1.0, new_temperature=1.0):
        with torch.no_grad():
            all_base_confidences, all_new_confidences, all_base_preds, all_new_preds = self._get_confidences(dataloader, base_temperature, new_temperature)

        meta_preds = []
        meta_choices = []

        for i in range(len(all_new_preds)):
            if all_base_confidences[i] > all_new_confidences[i]:
                meta_preds.append(all_base_preds[i])
                meta_choices.append(0)
            else:
                meta_preds.append(all_new_preds[i])
                meta_choices.append(1)

        meta_preds = np.array(meta_preds)
        meta_choices = np.array(meta_choices)

        return meta_preds, all_base_preds, all_new_preds, meta_choices

    def _get_confidences(self, dataloader, base_temperature, new_temperature):
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
        pass


def confidence(probs):
    return torch.max(probs, dim=1).values


combiners.register_builder("confidence", Confidence)
