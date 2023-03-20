import torch


class StackingDataset(torch.utils.data.Dataset):
    def __init__(self, logits_base, logits_new, labels):
        self.logits_base = logits_base
        self.logits_new = logits_new
        self.labels = labels

    def __len__(self):
        return len(self.logits_base)

    def __getitem__(self, index):
        logit_base, logit_new, label = self.logits_base[index], self.logits_new[index], self.labels[index]

        return logit_base, logit_new, label

