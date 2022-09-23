import torch

from .combiner import Combiner
from .creation import combiners


class KLDiv(Combiner):
    def __init__(self, base_model, new_model, dataset):
        super().__init__(base_model, new_model, dataset)

        self._setup()

    def predict(self, dataloader):
        all_base_preds = []
        all_new_preds = []

        all_base_kl_divs = []
        all_new_kl_divs = []

        for x, _, _, _ in dataloader:
            base_out = self.base_model(x)
            base_probs = torch.nn.Softmax(dim=1)(base_out)
            new_out = self.new_model(x)
            new_probs = torch.nn.Softmax(dim=1)(new_out)

            base_kl_divs = self._compute_kl_div(base_probs)
            new_kl_divs = self._compute_kl_div(new_probs)

            all_base_kl_divs.append(base_kl_divs)
            all_new_kl_divs.append(new_kl_divs)

            all_base_preds.append(torch.argmax(base_probs, dim=1))
            all_new_preds.append(torch.argmax(new_probs, dim=1))

        all_base_preds = torch.cat(all_base_preds)
        all_new_preds = torch.cat(all_new_preds)

        all_base_kl_divs = torch.cat(all_base_kl_divs)
        all_new_kl_divs = torch.cat(all_new_kl_divs)

        meta_preds = []

        for i in range(len(all_new_preds)):
            if all_base_kl_divs[i] > all_new_kl_divs[i]:
                meta_preds.append(all_base_preds[i])
            else:
                meta_preds.append(all_new_preds[i])

        meta_preds = torch.tensor(meta_preds)

        return meta_preds.cpu().numpy()

    def _compute_kl_div(self, probs):
        kl = kl_div(torch.ones_like(probs) / probs.shape[1], probs)

        return kl

    def _flatten(self, grad):
        sub_grads = []
        for g in grad:
            sub_grads.append(torch.flatten(g))

        return torch.cat(sub_grads).view(1, -1)

    def _setup(self):
        pass


def kl_div(a, b):
    return torch.sum(a * torch.log(a / b), dim=1)


combiners.register_builder("kl_div", KLDiv)
