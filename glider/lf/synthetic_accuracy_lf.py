from .labeling_function import SyntheticLabelingFunction
import numpy as np


class SyntheticAccuracyLF(SyntheticLabelingFunction):
    """
    LF with that returns the correct label with a given accuracy.
    """

    def __init__(self, accuracy, classes, name):
        self.accuracy = accuracy
        self.classes = np.array(list(classes))
        self.name = name

    def apply(self, context, label):

        if np.random.uniform(0, 1) <= self.accuracy:
            return label
        else:
            return np.random.choice(self.classes[self.classes != label], 1)[0]

    def get_name(self):
        return self.name
