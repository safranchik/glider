from torch.utils.data import Dataset
import numpy as np
import torch


class DataType(Dataset):
    """
    Abstract class defining the general properties of a dataset
    """

    def __init__(self, features, metadata=None):

        self.features = features

        self.metadata = metadata if metadata is not None else {}

        self.data_attributes = {}

    def __len__(self):
        return len(self.features)

    def __getitem__(self, ix):
        return self.features[ix]

    @property
    def classes(self):
        """
        :return: a set of unique classes defined in the weak label matrix and Y. In strongly labeled datasets,
            the weak label matrix will have no classes, and in weakly labeled datasets, Y will contain no classes.
        """
        raise NotImplementedError

    def class_to_id(self):
        """
        :return: Dictionary mapping class numerical indices to class names
        """
        return {class_name: i for (i, class_name) in enumerate(self.classes)}

    def sanitize_data_attr(self, attr, default_val=-1, dtype=np.float32):

        if self.features is not None:
            if attr is None:
                return default_val * np.ones(len(self.features))
            else:
                assert len(attr) == len(self.features)

        return np.array(attr, dtype=dtype)

    def add_data_attribute(self, name, attr):
        self.data_attributes[name] = self.sanitize_data_attr(attr)





