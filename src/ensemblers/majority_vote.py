import torch

from .creation import ensemblers
from .ensembler import Ensembler


class MajorityVote(Ensembler):
    def __init__(self, models):
        super().__init__(models)

    def predict(self, dataloader):
        all_preds = []

        with torch.no_grad():
            for x, _, _, _ in dataloader:

                model_preds = []
                for model in self.models:
                    out = model(x)
                    pred = torch.argmax(out, dim=1)

                    model_preds.append(pred)

                model_preds = torch.stack(model_preds, dim=1)
                all_preds.append(model_preds)

        all_preds = torch.cat(all_preds)

        return torch.mode(all_preds, dim=1)[0]


ensemblers.register_builder("majority_vote", MajorityVote)
