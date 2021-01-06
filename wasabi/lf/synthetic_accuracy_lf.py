from .labeling_function import LabelingFunction
import random


class SyntheticAccuracyLF(LabelingFunction):
    """
    LF with that returns the correct label with a given accuracy.
    """

    def __init__(self, accuracy, classes, name):
        self.accuracy = accuracy
        self.classes = set(classes)
        self.name = name

    def apply(self, sample, metadata):

        if random.uniform(0, 1) <= self.accuracy:
            return sample['Y']
        else:
            return random.sample(self.classes - {sample['Y']}, 1)[0]

    def get_name(self):
        return self.name
