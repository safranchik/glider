from .data_type import DataType
import numpy as np


class UnlabeledData(DataType):

    def __init__(self, features, metadata=None):
        DataType.__init__(self, features=features, metadata=metadata)

    def __add__(self, other):
        from .strong_data import StrongData

        if isinstance(other, UnlabeledData):
            return UnlabeledData(features=np.concatenate((self.features, other.features)),
                                 metadata=self.metadata.update(other.metadata))
        elif isinstance(other, StrongData):
            from .semi_supervised_data import SemiSupervisedData
            return SemiSupervisedData.from_dataset(self) + SemiSupervisedData.from_dataset(other)
        else:
            raise NotImplementedError

    @property
    def classes(self):
        return set()
