from torch.utils.data import Dataset


class DistillDatasetWrapper(Dataset):
    def __init__(self, dataset, distilled_knowledge=None):
        self.dataset = dataset
        self.distilled_knowledge = distilled_knowledge

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset.__getitem__(index)

        if self.distilled_knowledge is not None:
            knowledge = self.distilled_knowledge[index]
        else:
            knowledge = None

        return x, y, knowledge