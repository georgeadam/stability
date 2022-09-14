import torch

from .combiner import Combiner
from .creation import combiners


class Confidence(Combiner):
    def __init__(self, base_model, new_model, dataset):
        super().__init__(base_model, new_model, dataset)

        self._setup()

    def predict(self, dataloader):
        all_base_confidences = []
        all_new_confidences = []

        all_base_preds = []
        all_new_preds = []

        with torch.no_grad():
            for x, _, _, _ in dataloader:
                base_out = self.base_model(x)
                base_probs = torch.nn.Softmax(dim=1)(base_out)
                new_out = self.new_model(x)
                new_probs = torch.nn.Softmax(dim=1)(new_out)

                base_confidence = confidence(base_probs)
                new_confidence = confidence(new_probs)
                all_base_confidences.append(base_confidence)
                all_new_confidences.append(new_confidence)

                all_base_preds.append(torch.argmax(base_probs, dim=1))
                all_new_preds.append(torch.argmax(new_probs, dim=1))

        all_base_confidences = torch.cat(all_base_confidences)
        all_new_confidences = torch.cat(all_new_confidences)

        all_base_preds = torch.cat(all_base_preds)
        all_new_preds = torch.cat(all_new_preds)

        meta_preds = []

        for i in range(len(all_new_preds)):
            if all_base_confidences[i] > all_new_confidences[i]:
                meta_preds.append(all_base_preds[i])
            else:
                meta_preds.append(all_new_preds[i])

        meta_preds = torch.tensor(meta_preds)

        return meta_preds.cpu().numpy()

    def _setup(self):
        pass


def confidence(probs):
    return torch.max(probs, dim=1).values


combiners.register_builder("confidence", Confidence)
