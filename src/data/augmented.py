from torch.utils.data import Dataset


class AugmentedDataset(Dataset):
    def __init__(self, dataset, indices):
        super().__init__()

        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.dataset)

    @property
    def targets(self):
        return self.dataset.targets

    @targets.setter
    def targets(self, new_targets):
        self.dataset.targets = new_targets

    @property
    def data(self):
        return self.dataset.data

    @data.setter
    def data(self, new_data):
        self.dataset.data = new_data

    def __getitem__(self, idx):
        x, y = self.dataset.__getitem__(idx)

        return x, y, self.indices[idx]
