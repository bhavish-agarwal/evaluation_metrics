import numpy as np


def dcg_at_k(r, p, method=0):
    """
    As per https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Discounted_Cumulative_Gain
    :param r: relevance scores for the documents ordered by ranking algorithm
    :param p: DCG at rank position p
    :param method: Use 1 when stronger emphasis on retrieving relevant documents
    :return: DCG score at position p
    """
    r = np.asfarray(r)[:p]

    if r.size:
        if method == 0:
            return r[0] + np.sum(np.divide(r[1:], np.log2(np.arange(3, r.size + 2))))
        elif method == 1:
            return np.sum(np.divide(2**r-1, np.log2(np.arange(2, r.size + 2))))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, p, method=0):
    """
    As per https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
    :param r: relevance scores for the documents ordered by ranking algorithm
    :param p: NDCG at rank position p
    :param method: Use 1 when stronger emphasis on retrieving relevant documents
    :return: NDCG score at position p
    """
    ideal_dcg = dcg_at_k(sorted(r, reverse=True), p, method)

    if not ideal_dcg:
        return 0.

    return dcg_at_k(r, p, method)/ideal_dcg
