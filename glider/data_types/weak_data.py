from collections.abc import Iterable
import numpy as np
from ..lf import labels, weightings, LabelingFunction, SyntheticLabelingFunction
from .data_type import DataType
from .vote_matrix import VoteMatrix
from tqdm import tqdm


class WeakData(DataType):

    def __init__(self, features, weak_labels=None, vote_matrix=None, context=None, metadata=None,
                 weak_label_type="hard", weak_label_weighting="unweighted"):
        """
        :param features: input features.
        :param weak_labels: weak labels.
        :param vote_matrix: 2d matrix of votes.
        :param context: list of contextual information used context used to help users write labeling functions.
        :param metadata: metadata dictioanry for the given dataset.
        :param weak_label_type: one of the possible weak label types [hard/strong]
        :param weak_label_weighting: one of the possible weak label weightings. Specify "votes" to replace the
        weak labels with the votes from the weak label matrix.
        """

        DataType.__init__(self, features=features, metadata=metadata)

        if weak_labels is not None and vote_matrix is not None:
            raise AttributeError("You can only pass in a weak label array or a vote matrix to the WeakDatset class!")

        self.weak_labels = self.sanitize_data_attr(weak_labels, dtype=np.int)

        self.context = self.sanitize_data_attr(context, default_val=-1, dtype=np.object)

        # ensures weak labels are either 1d vectors, or 2d arrays with one soft label for each class
        if self.weak_labels.ndim == 2 and self.weak_labels.shape[1] != self.classes:
                self.weak_labels.flatten()

        self.vote_matrix = VoteMatrix(vote_matrix, len(self.features))

        # sets the weak label types (hard/soft) and weighting (e.g. softmax, unweighted)
        self.weak_label_type, self.weak_label_weighting = None, None
        self.set_weak_labels(weak_label_type=weak_label_type, weak_label_weighting=weak_label_weighting)
        self._update_Y_bar()

    def __getitem__(self, ix):
        return {"x": self.features[ix],
                "y_bar": self.weak_labels[ix],
                "index": ix}

    def __add__(self, other):

        if isinstance(other, WeakData):
            if self.weak_label_type != other.weak_label_type:
                raise AttributeError("Can only add weak datasets with the same label type.")

            if self.weak_label_weighting != other.weak_label_weighting:
                raise AttributeError("Can only add weak datasets with the same weak label weighting.")

            return WeakData(features=np.concatenate((self.features, other.features)),
                            weak_labels=np.concatenate((self.weak_labels, other.weak_labels)),
                            vote_matrix=None,  # TODO: concatenate vote matrices
                            context=np.concatenate((self.context, other.context)),
                            metadata=self.metadata.update(other.metadata),
                            weak_label_type=self.weak_label_type,
                            weak_label_weighting=self.weak_label_weighting)
        else:
            from .hybrid_data import HybridData
            return HybridData.from_dataset(self) + HybridData.from_dataset(other)

    @property
    def classes(self):
        # we discard the -1 class because it is associated to non-existent labels
        return set(np.unique(self.vote_matrix.votes)) | set(np.unique(self.weak_labels)) - {-1}

    def set_weak_labels(self, weak_label_type=None, weak_label_weighting=None):

        if weak_label_type:
            self.weak_label_type = getattr(labels, weak_label_type)
        if weak_label_weighting:
            self.weak_label_weighting = getattr(weightings, weak_label_weighting)

    def _update_Y_bar(self):
        """
        Computes the distribution of Y_bar for the given dataset
        :param weak_label_type: label type to use to compute the distribution of Y_bar
        :param in_place: whether to modify the dataset's Y_bar variable in place
        :return: Y_bar computed using the dataset's label type
        """

        if not len(self.vote_matrix.classes):
            return

        if self.weak_label_weighting == "votes":
            self.weak_labels = self.vote_matrix.votes
        else:
            label_distribution = self.weak_label_weighting(self.vote_matrix.votes, self.class_to_id())
            self.weak_labels = self.weak_label_type(label_distribution)

    def add_lf(self, lfs):
        """
        Adds a the votes from labeling function to the dataset..
        :param lfs: labeling function to apply to the weak dataset.
        :return: None.
        """

        for lf in self._sanitize_lfs(lfs):
            self.vote_matrix.add_lf(lf.get_name(), self.get_lf_votes(lf))

        self._update_Y_bar()

    def remove_lf(self, lfs):
        """
        Removes a the votes from labeling function that have been applied to the dataset.
        :param lfs: labeling function to remove from the weak dataset.
        :return: None.
        """

        for lf in self._sanitize_lfs(lfs):
            self.vote_matrix.remove_lf(lf.get_name())

        self._update_Y_bar()

    def update_lf(self, lfs):
        """
        Updates a the votes from labeling function that have been applied to the dataset.
        :param lfs: labeling function to update in the weak dataset.
        :return: None.
        """

        for lf in self._sanitize_lfs(lfs):
            self.vote_matrix.update_lf(lf.get_name(), self.get_lf_votes(lf))

        self._update_Y_bar()

    def get_lf_votes(self, lf):
        """
        Returns a list of votes for the given lf.
        :param lf: lf to apply to dataset.
        :return: list of votes applied to each feature sample.
        """
        if isinstance(lf, SyntheticLabelingFunction):
            raise ValueError("WeakDataset does not support synthetic labeling functions.")

        return [int(lf.apply(m)) for m in tqdm(self.context)]


    @staticmethod
    def _sanitize_lfs(lfs):
        if not isinstance(lfs, Iterable):
            lfs = [lfs]

        for lf in lfs:
            if not isinstance(lf, LabelingFunction) and not isinstance(lf, SyntheticLabelingFunction):
                raise ValueError(
                    'WeakDataset can only handle objects of type LabelingFunction, or iterables of such.')
        return lfs
