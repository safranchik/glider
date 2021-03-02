from .data_type import DataType
import numpy as np


class UnlabeledData(DataType):

    @property
    def classes(self):
        return set()

    def __add__(self, other):

        if isinstance(other, UnlabeledData):
            return UnlabeledData(features=np.concatenate((self.features, other.features)),
                                 metadata=self.metadata.update(other.metadata))

        else:
            from .hybrid_data import HybridData
            return HybridData.from_dataset(self) + HybridData.from_dataset(other)
