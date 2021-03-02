import numpy as np
from . import StrongData, WeakData, UnlabeledData
from ..lf import SyntheticLabelingFunction
from tqdm import tqdm


class HybridData(StrongData, WeakData):
    """
    Class defining the behavior of a fully labeled dataset
    """

    def __init__(self, features, labels=None, weak_labels=None, vote_matrix=None, strong_mask=None,
                 context=None, metadata=None, weak_label_type="hard", weak_label_weighting="unweighted"):
        """
        :param features: input features.
        :param labels: strong labels.
        :param weak_labels: weak labels.
        :param vote_matrix: 2d matrix of votes.
        :param context: list of contextual information used context used to help users write labeling functions.
        :param metadata: metadata dictionary for the given dataset.
        :param weak_label_type: one of the possible weak label types [hard/strong].
        :param weak_label_weighting: one of the possible weak label weightings. Specify "votes" to replace the
        weak labels with the votes from the weak label matrix.
        """

        # hybrid datasets inherit behavior from strong and weakly labeled datasets
        StrongData.__init__(self, features, labels)

        WeakData.__init__(self, features, weak_labels, vote_matrix, context, metadata,
                          weak_label_type, weak_label_weighting)

        self.strong_mask = self.sanitize_data_attr(strong_mask, dtype=np.bool)

        assert len(self.labels) == len(self.strong_mask) == len(self.features)

    def __getitem__(self, ix):
        return {"x": self.features[ix],
                "y": self.labels[ix],
                "y_bar": self.weak_labels[ix],
                "strong_mask": self.strong_mask[ix],
                "index": ix}

    def __add__(self, other):

        if not isinstance(other, HybridData):
            other = HybridData.from_dataset(other)

        if self.weak_label_type != other.weak_label_type:
            raise AttributeError("Can only add weak datasets with the same label type.")

        if self.weak_label_weighting != other.weak_label_weighting:
            raise AttributeError("Can only add weak datasets with the same weak label weighting.")

        if self.weak_labels.ndim != other.weak_labels.ndim:
            raise AttributeError("Dimensions of weak labels does not match: {} vs {}"
                                 .format(self.weak_labels.ndim, other.weak_labels.ndim))

        elif self.weak_labels.ndim == 2 and self.weak_labels.shape[1] != other.weak_labels.shape[1]:
            raise AttributeError("Shapes of weak labels does not match: {} vs {}"
                                 .format(self.weak_labels.shape[1], other.weak_labels.shape[1]))

        return HybridData(features=np.concatenate((self.features, other.features)),
                          labels=np.concatenate((self.labels, other.labels)),
                          weak_labels=np.concatenate((self.weak_labels, other.weak_labels)),
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
                       weak_labels=dataset.weak_labels,
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
