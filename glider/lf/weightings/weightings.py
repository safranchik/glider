import numpy as np
from scipy import special
from sklearn.preprocessing import normalize


def unweighted(votes, class_to_id):
    if votes.shape[1] == 0:
        return votes

    distribution = np.zeros((len(votes), len(class_to_id)))

    for index, row in enumerate(votes):
        for vote in row:
            if vote >= 0:
                distribution[index][class_to_id[vote]] += 1

    # uniformly increments the vote count of rows where where the LFs haven't cast votes,
    # so that we can normalize them to a uniform distribution
    no_votes = np.where(np.sum(distribution, axis=1) == 0)
    distribution[no_votes] += 1

    # returns normalized
    return normalize(distribution, norm='l1')


def softmax(votes, class_to_ix):

    if votes.shape[1] == 0:
        return None

    distribution = np.zeros((len(votes), len(class_to_ix)))

    for index, row in enumerate(votes):
        for vote in row:
            if vote >= 0:
                distribution[index][class_to_ix[vote]] += 1

    return special.softmax(distribution, axis=1)
