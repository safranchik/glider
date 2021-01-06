from torch.utils.data import Dataset
import torch
from .weak_label_matrix import WeakLabelMatrix
import numpy as np


class DataType(Dataset):
    """
    Abstract class defining the general properties of a dataset
    """

    def __init__(self, data_loader):
        """
        :param data_loader: loader for the dataset
        :param num_samples: number of samples to load from the data loader
        """

        loaded_data = data_loader.load(self.has_strong_labels(), self.has_weak_labels())
        self.X, self.Y, self.Y_bar, self.Lambda, self.metadata, self.featurizer_properties = loaded_data
        self.Lambda = WeakLabelMatrix(self.Lambda)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # returns dataset sample in the form of a dictionary
        return_dict = {'X': self.X[idx],
                       'Y': self.Y[idx],
                       'Y_bar': self.Y_bar[idx],
                       'Lambda': self.Lambda[idx],
                       'strong_label_mask': self.has_strong_labels(),
                       'weak_label_mask': self.has_weak_labels()}

        return return_dict

    def classes(self):
        """
        :return: a set of unique classes defined in the weak label matrix and Y. In strongly labeled datasets,
            the weak label matrix will have no classes, and in weakly labeled datasets, Y will contain no classes.
        """

        # we discard the -1 class (if exists) because it corresponds to abstained votes from the weak label matrix
        return set(np.unique(self.Y)) | self.Lambda.classes() - {-1}

    def class_to_ix(self):
        """
        :return: Dictionary mapping class numerical indices to class names
        """
        return {class_name: i for (i, class_name) in enumerate(self.classes())}

    def get_metadata(self):
        """
        :return: dataset metadata
        """
        return self.metadata

    def get_featurizer_properties(self):
        """
        :return: properties of the featurizer used in the dataset loader
        """
        return self.featurizer_properties

    def has_strong_labels(self):
        raise NotImplementedError()

    def has_weak_labels(self):
        raise NotImplementedError()
