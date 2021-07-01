import numpy as np
from . import StrongData, WeakData, UnlabeledData
from torch.utils.data import WeightedRandomSampler


class SemiWeakData(WeakData, UnlabeledData):
    """
    Class defining the behavior of a fully labeled dataset
    """

    def __init__(self, features, weak_mask=None, metadata=None, *args, **kwargs):
        """
        :param features: input features.
        :param labels: strong labels.
        :param context: list of contextual information used context used to help users write labeling functions.
        :param metadata: metadata dictionary for the given dataset.
        """

        # hybrid datasets inherit behavior from strong and weakly labeled datasets
        WeakData.__init__(self, features, metadata=metadata, *args, **kwargs)

        UnlabeledData.__init__(self, features, metadata=metadata)

        self.weak_mask = self.sanitize_data_attr(weak_mask, dtype=np.bool)

        assert len(self.targets) == len(self.weak_mask) == len(self.features)

    def __getitem__(self, ix):
        return {"x": self.features[ix],
                "y_bar": self.targets[ix],
                "weak_mask": self.weak_mask[ix],
                "index": ix}

    def __add__(self, other):

        if not isinstance(other, SemiWeakData):
            other = SemiWeakData.from_dataset(other)

        return SemiWeakData(features=np.concatenate((self.features, other.features)),
                            targets=np.concatenate((self.targets, other.targets)),
                            vote_matrix=None,
                            context=np.concatenate((self.context, other.context)),
                            weak_mask=np.concatenate((self.weak_mask, other.weak_mask)),
                            weak_label_type=self.weak_label_type.__name__,
                            weak_label_weighting=self.weak_label_weighting.__name__,
                            metadata=self.metadata.update(other.metadata))

    @property
    def classes(self):
        return super().classes | set(np.unique(self.targets)) - {-1}

    def set_sample_label_usage(self, idx, strong_label=True):
        """
        Updates the strong label mask for a given data sample.
        :param idx: dataset index to change label usage.
        :param strong_label: boolean indicating whether to use strong labels on this sample.
        :return: None.
        """
        self.weak_mask[idx] = strong_label

    @classmethod
    def from_dataset(cls, dataset):
        if isinstance(dataset, WeakData):
            return cls(dataset.features,
                       targets=dataset.targets,
                       vote_matrix=None,
                       context=dataset.context,
                       weak_label_type=dataset.weak_label_type.__name__,
                       weak_label_weighting=dataset.weak_label_weighting.__name__,
                       weak_mask=np.ones(len(dataset)),
                       metadata=dataset.metadata)

        elif isinstance(dataset, StrongData):
            return cls(dataset.features,
                       weak_mask=np.zeros(len(dataset)),
                       metadata=dataset.metadata)

        elif isinstance(dataset, UnlabeledData):
            return cls(dataset.features,
                       weak_mask=np.zeros(len(dataset)),
                       metadata=dataset.metadata)

    def get_weighted_random_sampler(self, weak_unlabeled_ratio):
        """
        Returns a weighted random sampler where strong and weak data samples are weighted are sampled equally.
        :return: weighted random sampler where strong and weak data have an equal probability of being sampled.
        """

        num_strong = sum(self.weak_mask)
        num_weak = sum(~self.weak_mask)

        weights = 1. / np.array([(1 + weak_unlabeled_ratio) * num_weak, (1 + 1 / weak_unlabeled_ratio) * num_strong])
        samples_weight = np.array([weights[int(m)] for m in self.weak_mask])

        assert np.isclose(sum(samples_weight), 1)

        return WeightedRandomSampler(samples_weight, len(samples_weight))
