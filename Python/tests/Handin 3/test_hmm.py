import unittest
import numpy as np

from hmm_jointprob import log_cost


class SoftmaxTestCase(unittest.TestCase):
    def test_softmax(self):
        X = np.array([[1, 2], [3, 4], [1, 2]])
        R = softmax(X)
        np.testing.assert_almost_equal(R[0, 0], 0.2689414213699951, 5)
        np.testing.assert_almost_equal(R[0, 1], 0.7310585786300049, 5)
