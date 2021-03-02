from .data_loader import DataLoader
import re
import numpy as np
import pandas as pd


class AGNewsLoader(DataLoader):

    def __init__(self, filename, featurizer=None, fit_featurizer=True, sections=None, dtype=np.float32):

        print('loading file {} from disk: '.format(filename), end='')
        self.df = pd.read_csv(filename, error_bad_lines=False, na_filter=False)
        print('done')

        DataLoader.__init__(self, filename, featurizer, fit_featurizer, sections, dtype)

    def get_input(self):
        return self.df['Title'] + ' ' + self.df['Description']

    def get_Y(self):
        return self.df['Class Index'].to_numpy(dtype=np.long) - 1  # subtract 1 because we want classes to start at 0

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
