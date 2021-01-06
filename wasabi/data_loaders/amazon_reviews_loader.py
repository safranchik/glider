from .data_loader import DataLoader
import re
import numpy as np
import pandas as pd


class AmazonReviewsPolarityLoader(DataLoader):

    def __init__(self, filename, featurizer, fit_featurizer=True, sections=None, dtype=np.float32, num_class_samples=150000):
        """
        :param filename:
        :param featurizer:
        :param dtype:
        :param max_samples: maximum number of samples of each rating class
        """

        print('Reading dataset from disk: ', end='')
        self.df = pd.read_csv(filename, sep='\t', error_bad_lines=False, warn_bad_lines=False, na_filter=False)
        print('done')

        self.indices = np.array([])
        for i in range(1, 6):
            # samples a class_sample number of samples with the given label
            class_samples = np.random.choice(np.where(self.df['star_rating'] == i)[0], size=num_class_samples, replace=False)
            self.indices = np.append(self.indices, class_samples)

        # shuffles indices for reviews so that they are not in ascending order of ratings
        np.random.shuffle(self.indices)

        # Initializes parent only after we've generated the values for our abstract functions
        DataLoader.__init__(self, filename, featurizer, fit_featurizer, sections, dtype)

    def get_input(self):
        return (self.df['review_headline'] + ' ' + self.df['review_body'])[self.indices]

    def get_Y(self):
        return np.array([1 if rating >= 4 else 0 for rating in self.df['star_rating'][self.indices]], dtype=np.int64)

    def get_Y_bar(self):
        return None

    def get_Lambda(self):
        return None

    def get_metadata(self):
        """
        :return: list of metadata dictionaries containing text and tokens
        """
        tokenizer_expression = r"[\w']+|[.,!?;]"
        return [{'text': body, 'tokens': re.findall(tokenizer_expression, body)} for body in self.get_input()]
