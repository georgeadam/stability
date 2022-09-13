import torch

from .combiner import Combiner
from .creation import combiners


class GradNorm(Combiner):
    def __init__(self, base_model, new_model, dataset):
        super().__init__(base_model, new_model, dataset)

        self._setup()

    def predict(self, dataloader):
        all_base_preds = []
        all_new_preds = []

        all_base_norms = []
        all_new_norms = []

        for x, _, _, _ in dataloader:
            base_out = self.base_model(x)
            base_probs = torch.nn.Softmax(dim=1)(base_out)
            new_out = self.new_model(x)
            new_probs = torch.nn.Softmax(dim=1)(new_out)

            base_norms = self._compute_grad_norms(base_probs, self.base_model)
            new_norms = self._compute_grad_norms(new_probs, self.new_model)

            all_base_norms.append(base_norms)
            all_new_norms.append(new_norms)

            all_base_preds.append(torch.argmax(base_probs, dim=1))
            all_new_preds.append(torch.argmax(new_probs, dim=1))

        all_base_preds = torch.cat(all_base_preds)
        all_new_preds = torch.cat(all_new_preds)

        all_base_norms = torch.cat(all_base_norms)
        all_new_norms = torch.cat(all_new_norms)

        meta_preds = []

        for i in range(len(all_new_preds)):
            if all_base_norms[i] > all_new_norms[i]:
                meta_preds.append(all_base_preds[i])
            else:
                meta_preds.append(all_new_preds[i])

        meta_preds = torch.tensor(meta_preds)

        return meta_preds.cpu().numpy()

    def _compute_grad_norms(self, probs, model):
        kl = kl_div(torch.ones_like(probs) / len(probs), probs)

        grads = [torch.autograd.grad(kl[i], model.parameters(), retain_graph=True) for i in range(len(kl))]
        grads = [self._flatten(grad) for grad in grads]
        grads = torch.cat(grads, dim=0)

        norms = torch.linalg.norm(grads, ord=1, dim=1)

        return norms

    def _flatten(self, grad):
        sub_grads = []
        for g in grad:
            sub_grads.append(torch.flatten(g))

        return torch.cat(sub_grads).view(1, -1)

    def _setup(self):
        pass


def kl_div(a, b):
    return torch.sum(a * torch.log(a / b), dim=1)


combiners.register_builder("gradnorm", GradNorm)
