import torch

from .creation import ensemblers
from .ensembler import Ensembler


class Average(Ensembler):
    def __init__(self, models, output_type):
        super().__init__(models)

        self.output_type = output_type

    def predict(self, dataloader):
        all_outs = []

        with torch.no_grad():
            for x, _, _, _ in dataloader:

                model_outs = []
                for model in self.models:
                    logits = model(x)
                    probs = torch.nn.Softmax(dim=1)(logits)

                    if self.output_type == "logit":
                        model_outs.append(logits)
                    elif self.output_type == "prob":
                        model_outs.append(probs)

                model_outs = torch.stack(model_outs, dim=1)
                all_outs.append(model_outs)


        all_outs = torch.cat(all_outs)
        all_outs = torch.mean(all_outs, dim=1)

        return torch.argmax(all_outs, dim=1)


ensemblers.register_builder("average", Average)