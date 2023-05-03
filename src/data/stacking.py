import torch


class StackingDataset(torch.utils.data.Dataset):
    def __init__(self, logits_base, logits_new, labels):
        self.logits_base = logits_base
        self.logits_new = logits_new
        self.labels = labels

        self.base_transform = None
        self.new_transform = None

    def __len__(self):
        return len(self.logits_base)

    def __getitem__(self, index):
        logit_base, logit_new, label = self.logits_base[index], self.logits_new[index], self.labels[index]

        if self.base_transform:
            logit_base = self.base_transform(logit_base)

        if self.new_transform:
            logit_new = self.new_transform(logit_new)

        return logit_base, logit_new, label

