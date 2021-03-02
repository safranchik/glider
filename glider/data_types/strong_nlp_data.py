import numpy as np
from .data_type import DataType


class StrongData(DataType):

    def __init__(self, features, labels=None, offsets=None, metadata=None):

        DataType.__init__(self, features=features, metadata=metadata)

        self.labels = self.sanitize_data_attr(labels, dtype=np.int)
        self.offsets = self.sanitize_data_attr(offsets, default_val=0, dtype=np.int)

    def __add__(self, other):

        if isinstance(other, StrongData):
            return StrongData(features=np.concatenate((self.features, other.features)),
                              labels=np.concatenate((self.labels, other.labels)),
                              metadata=self.metadata.update(other.metadata))

        else:
            from .hybrid_data import HybridData
            return HybridData.from_dataset(self) + HybridData.from_dataset(other)

    def __getitem__(self, ix):
        return {"x": self.features[ix],
                "y": self.labels[ix],
                "offsets": self.offsets[ix]}

    @property
    def classes(self):
        # we discard the -1 class because it is associated to non-existent labels
        return set(np.unique(self.labels)) - {-1}
