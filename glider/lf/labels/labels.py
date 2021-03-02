import numpy as np


def hard(label_distribution):
    """
    Converts an array of label distributions to an array of their modes (argmax values).
    Ties between labels with equal probability are resolved randomly
    :param label_distribution: numpy array containing label distributions for all samples in the dataset
    :return: modes of the label distributions
    """

    max_list = np.max(label_distribution, axis=1)
    return np.array([np.random.choice(np.where(dist == max_list[i])[0])
                     for i, dist in enumerate(label_distribution)], dtype=np.int64)


def soft(label_distribution):
    return label_distribution



