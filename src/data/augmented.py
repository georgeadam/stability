from torch.utils.data import Dataset


class AugmentedDataset(Dataset):
    def __init__(self, dataset, indices, source):
        super().__init__()

        self.dataset = dataset
        self.indices = indices
        self._source = source

    def __len__(self):
        return len(self.dataset)

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, new_source):
        self._source = new_source

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

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, new_transform):
        self.dataset.transform = new_transform

    def __getitem__(self, idx):
        x, y = self.dataset.__getitem__(idx)

        return x, y, self.indices[idx], self.source
