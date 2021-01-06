import pandas as pd
import numpy as np


class WeakLabelMatrix:
    """
    Class defining the properties of a weak label matrix.
    """

    def __init__(self, matrix):

        if not isinstance(matrix, pd.DataFrame):
            matrix = pd.DataFrame(matrix)

        self.matrix = matrix
        self.votes = self.matrix.values

    def __len__(self):
        return len(self.votes)

    def __getitem__(self, idx):
        return self.votes[idx]

    def __update_votes(self):
        """
        Updates the values of the LF votes, obtained from the DataFrame
        :return:
        """
        self.votes = self.matrix.values

    def add_lf(self, lf_name, lf_votes):
        """
        Adds a labeling function to the weak label matrix, and re-computes the np.array of votes.
        :param lf_name: name of labeling function to add.
        :param lf_votes: list of votes of the labeling function applied to the dataset.
        :return: None.
        """

        if lf_name in self.matrix:
            raise ValueError('LF name "%s" is already taken.' % lf_name)

        self.matrix[lf_name] = lf_votes
        self.__update_votes()

    def remove_lf(self, lf_name):
        """
        Removes a labeling function from the weak label matrix, and re-computes the np.array of votes.
        :param lf_name: name of labeling function to remove.
        :return: None.
        """

        if lf_name not in self.matrix:
            raise ValueError('The LF you are trying to remove does not exist.')

        del self.matrix[lf_name]
        self.__update_votes()

    def update_lf(self, lf_name, lf_votes):
        """
        Updates a labeling function from the weak label matrix.
        If the labeling function has already been added to the weak label matrix, it will be overwritten.
        Otherwise, it will be added.
        :param lf_name: name of labeling function to remove.
        :param lf_votes: list of votes of the labeling function applied to the dataset.
        :return: None.
        """

        self.matrix[lf_name] = lf_votes
        self.__update_votes()

    def get_num_lfs(self):
        """
        :return: number of labeling functions recorded in the weak label matrix.
        """
        return len(self.matrix.columns)

    def get_lf_names(self):
        """
        :return: name of labeling functions applied to the dataset.
        """
        return self.matrix.columns

    def get_lf_column_index(self, lf_name):
        """
        :param lf_name: name of labeling function to get column index of.
        :return: column index of labeling function as recorded in the weak label matrix.
        """
        return self.get_lf_names().get_loc(lf_name)

    def evaluate_lfs(self, labels):
        """
        Evaluates the labeling functions on the labeled data.
        :return: pd.DataFrame summary of the labeling functions
        """

        cols = ['Polarity', 'Correct', 'Incorrect', 'Accuracy', 'Coverage', 'Precision', 'Recall']
        summary = pd.DataFrame(columns=cols)

        for name, votes in zip(self.get_lf_names(), self.votes.T):
            correct_votes = votes == labels
            non_abstained_votes = votes >= 0

            polarity = np.unique(votes[non_abstained_votes])
            accuracy = np.mean(votes[non_abstained_votes] == labels[non_abstained_votes])
            coverage = sum(non_abstained_votes) / len(votes)

            tp = np.sum(correct_votes)
            fp = sum(non_abstained_votes) - tp
            tn = 0                  # true negative is zero in classification TODO: edit for other tasks
            fn = len(votes) - tp

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

            lf_row = pd.DataFrame(index=[name],
                                  data=[[polarity, tp, fp, accuracy, coverage, precision, recall]],
                                  columns=cols)

            summary = summary.append(lf_row)

        return summary

    def evaluate_predictions(self, labels, weighting, *args):

        cols = ['Correct', 'Incorrect', 'Accuracy']

        if not len(self.matrix.columns):
            return pd.DataFrame(columns=cols)

        np.random.seed(len(self.votes))     # reuse the same seed to replicate results

        label_distribution = self.get_label_distribution(weighting, *args)

        max_list = np.max(label_distribution, axis=1)
        predictions = np.array([np.random.choice(np.where(dist == max_list[i])[0])
                                for i, dist in enumerate(label_distribution)], dtype=np.int64)

        correct_predictions = (predictions == labels)

        correct = np.sum(correct_predictions)
        incorrect = len(self.votes) - np.sum(correct_predictions)
        accuracy = correct / len(self.votes)

        summary = pd.DataFrame(index=['Predictions'],
                               data=[[correct, incorrect, accuracy]],
                               columns=cols)

        return summary

    def get_label_distribution(self, weighting, *args):
        return weighting(self.votes, *args)

    def classes(self):
        return set(np.unique(self.votes)) - {-1}
