from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, data, targets):
        super().__init__()

        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, label = self.data[idx], self.targets[idx]

        return sentence, label
