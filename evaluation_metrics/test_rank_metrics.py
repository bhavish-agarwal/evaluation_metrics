import unittest
from rank_metrics import dcg_at_k, ndcg_at_k


class TestRankMetrics(unittest.TestCase):

    """
    Example from https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Example
    """
    def test_dcg(self):

        r = [3, 2, 3, 0, 1, 2]
        self.assertEqual(dcg_at_k(r, 1, method=0), 3)
        self.assertAlmostEqual( dcg_at_k(r, 6, method=0), 6.861, places=3)

    # verify two formulations of DCG are the same when the relevance values of
    # documents are binary.
    def test_two_dcg_for_binary_relevance(self):
        r = [1,1,0,0,1,1,0,1]

        self.assertEqual(dcg_at_k(r, 5, method=0), dcg_at_k(r, 5, method=1))
        self.assertEqual(dcg_at_k(r, 4, method=0), dcg_at_k(r, 4, method=1))

    def test_ndcg(self):
        r = [3, 2, 3, 0, 1, 2]

        # Already best ranked
        r_best = [5, 5, 5, 4, 4, 3]
        self.assertLessEqual(ndcg_at_k(r, 1,), 1.0)
        self.assertAlmostEqual(ndcg_at_k(r, 6, method=0), 0.961, places=3)
        self.assertLessEqual(ndcg_at_k(r_best, 4, ), 1.0)

    #Normalized DCG metric does not penalize for bad documents in the result.
    def test_limitation_1(self):
        r_one = [1, 1, 1]
        r_two = [1, 1, 1, 0]

        self.assertEqual(ndcg_at_k(r_one, 3, method=1), ndcg_at_k(r_two, 4, method=1))


if __name__ == '__main__':
    unittest.main()