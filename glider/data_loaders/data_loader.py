import pandas as pd
import numpy as np
from glider.featurizers import BagOfWords


class DataLoader:
    def __init__(self, filename, featurizer=None, fit_featurizer=True, sections=None, dtype=np.float32):
        """
        :param filename: name of file to load
        :param featurizer: used to extract features from input data
        :param fit_featurizer: boolean indicating whether to fit the featurizer to the given input
        :param sections: section limits used to partition dataset, if any (e.g. into train/test)
        :param dtype: data type of structures, used to ensure consistency
        """

        self.filename = filename
        self.featurizer = BagOfWords() if featurizer is None else featurizer
        self.dtype = dtype
        self.sections = sections

        if fit_featurizer:
            print('Fitting featurizer to data: ', end='')
            self.featurizer.fit(self.get_input())
            print('done')

        print('Applying featurizer to data: ', end='')
        self.X, self.featurizer_properties = self.featurizer.apply(self.get_input())
        print('done')

        self.Y = self.get_Y()
        self.Y_bar = self.get_Y_bar()
        self.Lambda = self.get_Lambda()
        self.metadata = self.get_metadata()

        self.validate_data()    # validates data structures returned by child subclass

        # splits data structures into sections if necessary
        if self.sections is not None:
            self.X = np.split(self.X, self.sections)
            self.Y = np.split(self.Y, self.sections)
            self.Y_bar = np.split(self.Y_bar, self.sections)
            self.Lambda = np.split(self.Lambda, self.sections)
            self.metadata = np.split(self.metadata, self.sections)

        self.current_section = 0

    def validate_data(self):
        """
        Ensures consistent dimensions between existing data structures, initializing as necessary
        :return: None
        """
        if self.X is None:
            raise ValueError('You need to specify input values.')

        if self.Y is None:
            self.Y = np.full((len(self.X),), -1, dtype=self.dtype)
        elif len(self.Y) != len(self.X):
            raise ValueError('Dimensions for input and output labels must match.')

        if self.Y_bar is None:
            self.Y_bar = np.array([[] for _ in range(len(self.X))], dtype=self.dtype)
        elif len(self.Y_bar) != len(self.X):
            raise ValueError('Dimensions for input and proxy output labels must match.')

        if self.Lambda is None:
            self.Lambda = pd.DataFrame(index=np.arange(len(self.X)), columns=[], dtype=self.dtype)
        elif len(self.Lambda) != len(self.X):
            raise ValueError('Dimensions for input and label matrix must match.')

        if self.metadata is None:
            self.metadata = np.empty(len(self.X), dtype=object)
        elif len(self.metadata) != len(self.X):
                raise ValueError('Dimensions for input and metadata must match.')

    def load(self, has_strong_labels, has_weak_labels):
        """
        Returns data structures for a a given dataset, modifying as necessry
        :param has_strong_labels: boolean indicating whether the dataset contains strong labels
        :param has_weak_labels: boolean indicating whether the dataset contains weak labels
        :return: X, Y, Y_bar, Lambda, metadata for dataset
        """

        if self.current_section >= len(self.X):
            raise ValueError('Can only load dataset in {} partition(s)'.format(len(self.X)))

        if self.sections is not None:
            X = self.X[self.current_section]
            Y = self.Y[self.current_section]
            Y_bar = self.Y_bar[self.current_section]
            Lambda = self.Lambda[self.current_section]
            metadata = self.metadata[self.current_section]
            self.current_section += 1
        else:
            X = self.X
            Y = self.Y
            Y_bar = self.Y_bar
            Lambda = self.Lambda
            metadata = self.metadata

        if not has_strong_labels:
            Y = np.full((len(X),), -1, dtype=self.dtype)

        if not has_weak_labels:
            Y_bar = np.array([[] for _ in range(len(X))], dtype=self.dtype)
            Lambda = pd.DataFrame(index=np.arange(len(X)), columns=[], dtype=self.dtype)

        return X, Y, Y_bar, Lambda, metadata, self.featurizer_properties

    def get_input(self):
        raise NotImplementedError

    def get_Y(self):
        raise NotImplementedError

    def get_Y_bar(self):
        raise NotImplementedError

    def get_Lambda(self):
        raise NotImplementedError

    def get_metadata(self):
        raise NotImplementedError

    def get_num_samples(self):
        return len(self.X)

    def get_featurizer_properties(self):
        return self.featurizer_properties

    def reset_sections(self):
        self.current_section = 0
