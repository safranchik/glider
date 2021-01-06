from .data_type import DataType


class StronglyLabeledDataset(DataType):

    def __init__(self, data_loader):
        DataType.__init__(self, data_loader)

    def has_strong_labels(self):
        return True

    def has_weak_labels(self):
        return False
