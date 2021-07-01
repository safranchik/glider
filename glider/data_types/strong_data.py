import numpy as np
from .data_type import DataType
from typing import Iterable


class StrongData(DataType):

    def __init__(self, features, targets=None, metadata=None):

        DataType.__init__(self, features=features, metadata=metadata)

        self.labels = self.sanitize_data_attr(targets, dtype=np.long).reshape(-1)

    def __add__(self, other: DataType):
        from .unlabeled_data import UnlabeledData
        from .weak_data import WeakData

        if isinstance(other, StrongData):
            return StrongData(features=np.concatenate((self.features, other.features)),
                              targets=np.concatenate((self.labels, other.labels)),
                              metadata=self.metadata.update(other.metadata))

        elif isinstance(other, WeakData):
            from .inter_supervised_data import InterSupervisedData
            return InterSupervisedData.from_dataset(self) + InterSupervisedData.from_dataset(other)
        elif isinstance(other, UnlabeledData):
            from .semi_supervised_data import SemiSupervisedData
            return SemiSupervisedData.from_dataset(self) + SemiSupervisedData.from_dataset(other)
        else:
            raise NotImplementedError

    def __getitem__(self, ix):
        return self.features[ix], self.labels[ix]

    @property
    def classes(self):
        # we discard the -1 class because it is associated to non-existent labels
        return set(np.unique(self.labels)) - {-1}

    def partition(self, partition_indices: Iterable[Iterable[int]]):
        new_datasets = []

        for indices in partition_indices:
            new_datasets.append(StrongData(features=self.features[indices],
                                           targets=self.labels[indices],
                                           metadata=self.metadata))

        return new_datasets

    def relabel_strong(self, ix, label: int):
        self.labels[ix] = label


