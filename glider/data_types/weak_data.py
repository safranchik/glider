from collections.abc import Iterable
import numpy as np
from ..lf import weightings, LabelingFunction, SyntheticLabelingFunction
from .data_type import DataType
from .vote_matrix import VoteMatrix
from tqdm import tqdm
from typing import Iterable
from .util import soften, harden
import torch

class WeakData(DataType):

    def __init__(self, targets=None, vote_matrix=None, context=None,
                 weak_label_type="soft", weak_label_weighting="unweighted", *args, **kwargs):
        """
        :param targets: weak labels.
        :param vote_matrix: 2d matrix of votes.
        :param context: list of contextual information used context used to help users write labeling functions.
        :param metadata: metadata dictioanry for the given dataset.
        :param weak_label_type: one of the possible weak label types [hard/strong]
        :param weak_label_weighting: one of the possible weak label weightings. Specify "votes" to replace the
        weak labels with the votes from the weak label matrix.
        """

        DataType.__init__(self, *args, **kwargs)

        if targets is not None and vote_matrix is not None:
            raise AttributeError("You can only pass in a weak label array or a vote matrix to the WeakDatset class!")

        self.targets = self.sanitize_data_attr(targets, dtype=np.long)
        self.context = self.sanitize_data_attr(context, default_val=-1)
        self.vote_matrix = VoteMatrix(vote_matrix, len(self.features))
        self.weak_label_type = weak_label_type

        # sets the weak label types (hard/soft) and weighting (e.g. softmax, unweighted)
        self.weak_label_weighting = getattr(weightings, weak_label_weighting)

        # ensures weak labels are either 1d vectors, or 2d arrays with one soft label for each class
        if self.targets.ndim == 2:
            if self.targets.shape[1] != self.classes:
                self.targets.flatten()

        if self.weak_label_type == "hard":
            self.targets = harden(self.targets)
        elif self.weak_label_type == "soft":
            self.targets = soften(self.targets, len(self.classes))

        self._update_Y_bar()

    def __getitem__(self, ix):
        return self.features[ix], self.targets[ix]

    def __add__(self, other):
        from .unlabeled_data import UnlabeledData
        from .strong_data import StrongData

        if isinstance(other, WeakData):
            if self.weak_label_type != other.weak_label_type:
                raise AttributeError("Can only add weak datasets with the same label type.")

            if self.weak_label_weighting != other.weak_label_weighting:
                raise AttributeError("Can only add weak datasets with the same weak label weighting.")

            return WeakData(features=np.concatenate((self.features, other.features)),
                            targets=np.concatenate((self.targets, other.targets)),
                            vote_matrix=None,  # TODO: concatenate vote matrices
                            context=np.concatenate((self.context, other.context)),
                            weak_label_type=self.weak_label_type.__name__,
                            weak_label_weighting=self.weak_label_weighting.__name__,
                            metadata=self.metadata.update(other.metadata))

        elif isinstance(other, StrongData):
            from .inter_supervised_data import InterSupervisedData
            return InterSupervisedData.from_dataset(self) + InterSupervisedData.from_dataset(other)
        elif isinstance(other, UnlabeledData):
            from .semi_weak_data import SemiWeakData
            return SemiWeakData.from_dataset(self) + SemiWeakData.from_dataset(other)
        else:
            raise NotImplementedError

    @property
    def classes(self):
        # we discard the -1 class because it is associated to non-existent labels
        return set(np.unique(self.vote_matrix.votes)) | set(np.unique(self.targets)) - {-1}

    def relabel_weak(self, ix, weak_label):
        if type(self.targets[ix]) is np.ndarray:
            self.targets[ix] = np.array(weak_label)
        else:
            self.targets[ix] = weak_label

    def partition(self, partition_indices: Iterable[Iterable[int]]):
        new_datasets = []

        for indices in partition_indices:
            new_datasets.append(WeakData(features=self.features[indices],
                                         targets=self.targets[indices],
                                         context=self.context[indices],
                                         weak_label_type=self.weak_label_type.__name__,
                                         weak_label_weighting=self.weak_label_weighting.__name__,
                                         metadata=self.metadata))

        return new_datasets

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
            self.targets = self.vote_matrix.votes
        else:
            label_distribution = self.weak_label_weighting(self.vote_matrix.votes, self.class_to_id())
            # self.targets = self.weak_label_type(label_distribution)

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
