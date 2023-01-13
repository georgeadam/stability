import torch.nn

from .creation import losses
from .loss import LossInterface


class SpecificCELoss(LossInterface):
    def __call__(self, out, y):
        ce_loss = torch.nn.CrossEntropyLoss()(out, y)
        second_likeliest = torch.topk(out, dim=1, k=2).indices
        second_likeliest = second_likeliest[:, 1]
        with torch.no_grad():
            relevant_samples = (second_likeliest != y).float()

        probs = torch.nn.LogSoftmax(dim=1)(out)
        probs = 1 - probs

        second_likeliest_loss = torch.nn.NLLLoss(reduction="none")(probs, second_likeliest)
        second_likeliest_loss *= relevant_samples
        second_likeliest_loss = second_likeliest_loss.sum()

        return ce_loss +  second_likeliest_loss


losses.register_builder("specific_ce", SpecificCELoss)