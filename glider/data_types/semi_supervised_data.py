import numpy as np
from . import StrongData, WeakData, UnlabeledData
from torch.utils.data import WeightedRandomSampler


class SemiSupervisedData(StrongData, UnlabeledData):
    """
    Class defining the behavior of a fully labeled dataset
    """

    def __init__(self, features, labels=None, strong_mask=None, metadata=None):
        """
        :param features: input features.
        :param labels: strong labels.
        :param context: list of contextual information used context used to help users write labeling functions.
        :param metadata: metadata dictionary for the given dataset.
        """

        # hybrid datasets inherit behavior from strong and weakly labeled datasets
        StrongData.__init__(self, features, labels)

        UnlabeledData.__init__(self, features, metadata)

        self.strong_mask = self.sanitize_data_attr(strong_mask, dtype=np.bool)

        assert len(self.labels) == len(self.strong_mask) == len(self.features)

    def __getitem__(self, ix):
        return {"x": self.features[ix],
                "y": self.labels[ix],
                "strong_mask": self.strong_mask[ix],
                "index": ix}

    def __add__(self, other):

        if not isinstance(other, SemiSupervisedData):
            other = SemiSupervisedData.from_dataset(other)

        return SemiSupervisedData(features=np.concatenate((self.features, other.features)),
                                  labels=np.concatenate((self.labels, other.labels)),
                                  strong_mask=np.concatenate((self.strong_mask, other.strong_mask)),
                                  metadata=self.metadata.update(other.metadata))

    @property
    def classes(self):
        return super().classes | set(np.unique(self.labels)) - {-1}

    def set_sample_label_usage(self, idx, strong_label=True):
        """
        Updates the strong label mask for a given data sample.
        :param idx: dataset index to change label usage.
        :param strong_label: boolean indicating whether to use strong labels on this sample.
        :return: None.
        """
        self.strong_mask[idx] = strong_label

    @classmethod
    def from_dataset(cls, dataset):
        if isinstance(dataset, WeakData):
            return cls(dataset.features,
                       strong_mask=np.zeros(len(dataset)),
                       metadata=dataset.metadata)

        elif isinstance(dataset, StrongData):
            return cls(dataset.features,
                       labels=dataset.labels,
                       strong_mask=np.ones(len(dataset)),
                       metadata=dataset.metadata)

        elif isinstance(dataset, UnlabeledData):
            return cls(dataset.features,
                       strong_mask=np.zeros(len(dataset)),
                       metadata=dataset.metadata)

    def get_weighted_random_sampler(self):
        """
        Returns a weighted random sampler where strong and weak data samples are weighted are sampled equally.
        :return: weighted random sampler where strong and weak data have an equal probability of being sampled.
        """

        weights = 1. / np.array([sum(~self.strong_mask), sum(self.strong_mask)])
        samples_weight = np.array([weights[int(m)] for m in self.strong_mask])

        return WeightedRandomSampler(samples_weight, len(samples_weight))
