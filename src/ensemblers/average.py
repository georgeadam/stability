import torch

from .creation import ensemblers
from .ensembler import Ensembler


class Average(Ensembler):
    def __init__(self, models):
        super().__init__(models)

    def predict(self, dataloader):
        all_outs = []

        with torch.no_grad():
            for x, _, _, _ in dataloader:

                model_outs = []
                for model in self.models:
                    out = model(x)

                    model_outs.append(out)

                model_outs = torch.stack(model_outs, dim=1)
                all_outs.append(model_outs)


        all_outs = torch.cat(all_outs)
        all_outs = torch.mean(all_outs, dim=1)

        return torch.argmax(all_outs, dim=1)


ensemblers.register_builder("average", Average)