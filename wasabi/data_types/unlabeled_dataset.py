from torch.utils.data import Dataset
import torch


class UnlabeledDataset(Dataset):

    def __init__(self):
        self.metadata = []
        self.properties = {}

        self.X = self.load_data()

    def load_data(self, *args):
        raise NotImplementedError

    def __len__(self):

        if self.X is None:
            raise ValueError(
                'LabeledDataset is an abstract class and should not be instantiated.')

        return len(self.X)

    def __getitem__(self, idx):

        if self.X is None:
            raise ValueError(
                'LabeledDataset is an abstract class and should not be instantiated.')

        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx]

    def get_metadata(self):
        return self.metadata
