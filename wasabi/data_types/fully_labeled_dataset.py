import numpy as np
from .weakly_labeled_dataset import WeaklyLabeledDataset
import torch


class FullyLabeledDataset(WeaklyLabeledDataset):
    """
    Class defining the behavior of a fully labeled dataset
    """

    def __init__(self, data_loader, weak_label_type='soft', weak_label_weighting='unweighted'):
        """
        :param data_loader: loader for the dataset
        :param num_samples: number of samples to load from the data loader
        :param weighting: labeling function weighting, used to aggregate votes
        :param weak_label_type: indicates default label type for dataset
        """
        # fully labeled datasets inherit behavior from weakly labeled datasets
        WeaklyLabeledDataset.__init__(self, data_loader, weak_label_type, weak_label_weighting)

        self.strong_labels = np.ones(len(self.X), dtype=bool)
        self.weak_labels = np.ones(len(self.X), dtype=bool)

    def __getitem__(self, idx):
        """
        Overrides default DataType method to return sample-specific label masks
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return_dict = {'X': self.X[idx],
                       'Y': self.Y[idx],
                       'Y_bar': self.Y_bar[idx],
                       'Lambda': self.Lambda[idx],
                       'strong_label_mask': self.strong_labels[idx],    # sample-specific strong label mask
                       'weak_label_mask': self.weak_labels[idx]}        # sample-specific weak label mask

        return return_dict

    def evaluate_lfs(self):
        return self.Lambda.evaluate_lfs(self.Y)

    def evaluate_predictions(self):
        """
        Evaluates the aggregate predictions of the labeling functions using either hard labels or the class-specific
        weighting method
        :return: empirical accuracy of the combined labeling functions
        """

        return self.Lambda.evaluate_predictions(self.Y, self.weak_label_weighting, self.X, self.metadata, self.class_to_ix())

    def set_sample_label_usage(self, idx, strong_label=True, weak_label=True):
        """
        Sets the label usage for a dataset sample indexed by the variable idx
        :param idx: dataset index to change label usage
        :param strong_label: boolean indicating whether to use strong labels on this sample
        :param weak_label: boolean indicating whether to use strong labels on this sample
        :return: None
        """
        self.strong_labels[idx] = strong_label
        self.weak_labels[idx] = weak_label

    def has_strong_labels(self):
        return True

    def has_weak_labels(self):
        return True
