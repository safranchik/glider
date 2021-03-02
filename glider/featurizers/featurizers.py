from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


class Featurizer:

    def apply(self, X):
        return self._get_features(X), self._get_properties()

    def fit(self, data):
        raise NotImplementedError

    def _get_features(self, X):
        raise NotImplementedError

    def _get_properties(self):
        raise NotImplementedError


class BagOfWords(Featurizer):
    def __init__(self, num_features=5000, token_pattern=None, dtype=np.float32):
        self.num_features = num_features
        self.token_pattern = token_pattern
        self.dtype = dtype

        if token_pattern is None:
            self.vectorizer = CountVectorizer(max_features=self.num_features, dtype=dtype)
        else:
            self.vectorizer = CountVectorizer(max_features=self.num_features, token_pattern=token_pattern, dtype=dtype)

    def fit(self, data):
        self.vectorizer.fit(data)

    def _get_features(self, data):
        return self.vectorizer.transform(data).toarray()

    def _get_properties(self):
        return {'type': 'bag_of_words',
                'num_features': self.num_features,
                'token_pattern': self.vectorizer.token_pattern,
                'vocabulary': self.vectorizer.vocabulary_,
                'dtype': self.dtype}


class BagOfNgrams(Featurizer):
    def __init__(self, num_features=5000, ngram_range=(1, 2), token_pattern=None, dtype=np.float32):
        self.num_features = num_features
        self.token_pattern = token_pattern
        self.ngram_range = ngram_range
        self.dtype = dtype


        if token_pattern is None:
            self.vectorizer = CountVectorizer(ngram_range=self.ngram_range,
                                              max_features=self.num_features,
                                              dtype=np.float32)
        else:
            self.vectorizer = CountVectorizer(ngram_range=self.ngram_range,
                                              max_features=self.num_features,
                                              token_pattern=token_pattern,
                                              dtype=np.float32)

    def fit(self, data):
        self.vectorizer.fit(data)

    def _get_features(self, data):
        return self.vectorizer.transform(data).toarray()

    def _get_properties(self):
        return {'type': 'bag_of_words',
                'num_features': self.num_features,
                'ngram_range': self.ngram_range,
                'token_pattern': self.vectorizer.token_pattern,
                'vocabulary': self.vectorizer.vocabulary_,
                'dtype': self.dtype}

