import numpy as np
from . import StrongData, WeakData, UnlabeledData
from ..lf import SyntheticLabelingFunction
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
from typing import TypeVar, Generic, Iterable, Iterator, Sequence, List, Optional, Tuple, Union


class InterSupervisedData(StrongData, WeakData):
    """
    Class defining the behavior of a fully labeled dataset
    """

    def __init__(self, features, labels=None, targets=None, vote_matrix=None, strong_mask=None,
                 context=None, metadata=None, weak_label_type="hard", weak_label_weighting="unweighted"):
        """
        :param features: input features.
        :param labels: strong labels.
        :param targets: weak labels.
        :param vote_matrix: 2d matrix of votes.
        :param context: list of contextual information used context used to help users write labeling functions.
        :param metadata: metadata dictionary for the given dataset.
        :param weak_label_type: one of the possible weak label types [hard/strong].
        :param weak_label_weighting: one of the possible weak label weightings. Specify "votes" to replace the
        weak labels with the votes from the weak label matrix.
        """

        # hybrid datasets inherit behavior from strong and weakly labeled datasets
        StrongData.__init__(self, features, labels)

        WeakData.__init__(self, features, targets, vote_matrix, context, metadata,
                          weak_label_type, weak_label_weighting)

        self.strong_mask = self.sanitize_data_attr(strong_mask, dtype=np.bool)

        assert len(self.targets) == len(self.strong_mask) == len(self.features)

    def __getitem__(self, ix):
        return {"x": self.features[ix],
                "y": self.labels[ix],
                "y_bar": self.targets[ix],
                "strong_mask": self.strong_mask[ix],
                "index": ix}

    def __add__(self, other):
        if not isinstance(other, InterSupervisedData):
            other = InterSupervisedData.from_dataset(other)

        if self.targets.ndim != other.targets.ndim:

            if self.targets.ndim == 1:
                self.soften_labels()
            elif other.targets.ndim == 1:
                other.soften_labels()
            else:
                raise AttributeError("Dimensions of weak labels does not match: {} vs {}"
                                     .format(self.targets.ndim, other.targets.ndim))

        elif self.targets.ndim == 2 and self.targets.shape[1] != other.targets.shape[1]:
            Warning("Hardening weak labels of shape {} and {}"
                    .format(self.targets.shape[1], other.targets.shape[1]))

        return InterSupervisedData(features=np.concatenate((self.features, other.features)),
                                   labels=np.concatenate((self.labels, other.labels)),
                                   targets=np.concatenate((self.targets, other.targets)),
                                   vote_matrix=None,  # TODO: concatenate vote matrices
                                   strong_mask=np.concatenate((self.strong_mask, other.strong_mask)),
                                   context=np.concatenate((self.context, other.context)),
                                   metadata=self.metadata.update(other.metadata),
                                   weak_label_type=self.weak_label_type.__name__,
                                   weak_label_weighting=self.weak_label_weighting.__name__)

    @property
    def classes(self):
        return super().classes | set(np.unique(self.labels)) - {-1}

    def evaluate_lfs(self):
        return self.vote_matrix.evaluate_lfs(labels=self.labels)

    def evaluate_predictions(self):
        """
        Evaluates the aggregate predictions of the labeling functions using either hard labels or the class-specific
        weighting method
        :return: empirical accuracy of the combined labeling functions
        """

        return self.vote_matrix.evaluate_predictions(labels=self.labels,
                                                     weighting=self.weak_label_weighting,
                                                     class_to_id=self.class_to_id())

    def set_sample_label_usage(self, idx, strong_label=True):
        """
        Updates the strong label mask for a given data sample.
        :param idx: dataset index to change label usage.
        :param strong_label: boolean indicating whether to use strong labels on this sample.
        :return: None.
        """
        self.strong_mask[idx] = strong_label

    def get_lf_votes(self, lf):
        """
        Returns a list of votes for the given lf.
        :param lf: lf to apply to dataset.
        :return: list of votes applied to each feature sample.
        """

        if isinstance(lf, SyntheticLabelingFunction):
            return [int(lf.apply(m, l)) for m, l in tqdm(zip(self.context, self.labels))]
        return [int(lf.apply(m)) for m in tqdm(self.context)]

    @classmethod
    def from_dataset(cls, dataset):
        if isinstance(dataset, WeakData):
            return cls(dataset.features,
                       targets=dataset.targets,
                       vote_matrix=None,
                       strong_mask=np.zeros(len(dataset)),
                       metadata=dataset.metadata,
                       weak_label_type=dataset.weak_label_type.__name__,
                       weak_label_weighting=dataset.weak_label_weighting.__name__)

        elif isinstance(dataset, StrongData):
            return cls(dataset.features,
                       labels=dataset.labels,
                       strong_mask=np.ones(len(dataset)),
                       metadata=dataset.metadata)

        elif isinstance(dataset, UnlabeledData):
            return cls(dataset.features,
                       strong_mask=np.zeros(len(dataset)),
                       metadata=dataset.metadata)

    def get_weighted_random_sampler(self, strong_weak_ratio=1):
        """
        Returns a weighted random sampler where strong and weak data samples are weighted are sampled equally.
        :return: weighted random sampler where strong and weak data have an equal probability of being sampled.
        """

        num_strong = sum(self.strong_mask)
        num_weak = sum(~self.strong_mask)

        weights = 1. / np.array([(1+strong_weak_ratio) * num_weak, (1 + 1/strong_weak_ratio) * num_strong])
        samples_weight = np.array([weights[int(m)] for m in self.strong_mask])

        assert np.isclose(sum(samples_weight), 1)

        return WeightedRandomSampler(samples_weight, len(samples_weight))

    def partition(self, partition_indices: Iterable[Iterable[int]]):
        new_datasets = []

        for indices in partition_indices:
            new_datasets.append(InterSupervisedData(features=self.features[indices],
                                                    labels=self.labels[indices],
                                                    targets=self.targets[indices],
                                                    vote_matrix=None,
                                                    strong_mask=self.strong_mask[indices],
                                                    context=self.context[indices],
                                                    metadata=self.metadata,
                                                    weak_label_type=self.weak_label_type.__name__,
                                                    weak_label_weighting=self.weak_label_weighting.__name__))

        return new_datasets if len(new_datasets) > 1 else new_datasets[0]
