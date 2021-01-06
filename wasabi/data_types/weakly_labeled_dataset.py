from .data_type import DataType
from collections.abc import Iterable
from tqdm import tqdm
from ..lf import labels, weightings, LabelingFunction


class WeaklyLabeledDataset(DataType):

    def __init__(self, data_loader, weak_label_type='soft', weak_label_weighting='unweighted'):
        DataType.__init__(self, data_loader)

        self.weak_label_type = None
        self.weak_label_weighting = None

        self.set_weak_labels(weak_label_type=weak_label_type, weak_label_weighting=weak_label_weighting)

    def add_lf(self, lfs):

        lfs = self._sanitize_lfs(lfs)

        for lf in lfs:
            self.Lambda.add_lf(lf.get_name(), self.get_lf_votes(lf))

        self._update_Y_bar()

    def remove_lf(self, lfs):

        lfs = self._sanitize_lfs(lfs)

        for lf in lfs:
            self.Lambda.remove_lf(lf.get_name())

        self._update_Y_bar()

    def update_lf(self, lfs):

        lfs = self._sanitize_lfs(lfs)

        for lf in lfs:
            self.Lambda.update_lf(lf.get_name(), self.get_lf_votes(lf))

        self._update_Y_bar()

    def get_lf_votes(self, lf):
        """
        Returns a list of votes for the given lf
        :param lf: lf to apply to dataset
        :return: list of votes applied to each sample in X
        """
        return [lf.get_vote(self.__getitem__(i), self.metadata[i]) for i in tqdm(range(len(self)))]

    def set_weak_labels(self, weak_label_type='none', weak_label_weighting='soft'):

        if weak_label_type == self.weak_label_type:
            return

        self.weak_label_weighting = getattr(weightings, weak_label_weighting)
        self.weak_label_type = getattr(labels, weak_label_type)
        self._update_Y_bar()

    def _update_Y_bar(self, in_place=True):
        """
        Computes the distribution of Y_bar for the given dataset
        :param weak_label_type: label type to use to compute the distribution of Y_bar
        :param in_place: whether to modify the dataset's Y_bar variable in place
        :return: Y_bar computed using the dataset's label type
        """

        label_distribution = self.weak_label_weighting(self.X, self.Lambda.votes, self.class_to_ix(), self.metadata)
        Y_bar = self.weak_label_type(label_distribution)

        if in_place:
            self.Y_bar = Y_bar

        return Y_bar

    def has_strong_labels(self):
        return False

    def has_weak_labels(self):
        return True

    @staticmethod
    def _sanitize_lfs(lfs):
        if not isinstance(lfs, Iterable):
            if not isinstance(lfs, LabelingFunction):
                raise ValueError(
                    'WeakDataset can only handle objects of type LabelingFunction, or iterables of such.')
            lfs = [lfs]

        for lf in lfs:
            if not isinstance(lf, LabelingFunction):
                raise ValueError(
                    'WeakDataset can only handle objects of type LabelingFunction, or iterables of such.')
        return lfs
