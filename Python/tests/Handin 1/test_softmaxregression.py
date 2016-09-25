import unittest
import numpy as np
from softmaxregression import softmax, soft_cost, batch_grad_descent, mini_batch_grad_descent


class SoftmaxTestCase(unittest.TestCase):
    def test_softmax(self):
        X = np.array([[1, 2], [3, 4], [1, 2]])
        R = softmax(X)
        np.testing.assert_almost_equal(R[0, 0], 0.2689414213699951, 5)
        np.testing.assert_almost_equal(R[0, 1], 0.7310585786300049, 5)
        np.testing.assert_almost_equal(R[1, 0], 0.2689414213699951, 5)
        np.testing.assert_almost_equal(R[1, 1], 0.73105857863000479, 5)
        np.testing.assert_almost_equal(R[2, 0], 0.2689414213699951, 5)
        np.testing.assert_almost_equal(R[2, 1], 0.7310585786300049, 5)


class SoftCostTestCase(unittest.TestCase):
    def test_softcost(self):
        # d=2, n=3, K=4
        X = np.array([[1, 2],
                      [3, 4],
                      [5, 6]])
        Y = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0]])
        W = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8]])
        cost, grad = soft_cost(X, Y, W, reg=0)
        np.testing.assert_almost_equal(cost, 0.00013479850454897337, 7)
        np.testing.assert_almost_equal(grad.shape, W.shape, 7)

class BatchGradDescentTestCase(unittest.TestCase):
    def test_batch_grad_descent(self):
        # d=2, n=3, K=4
        X = np.array([[1, 2],
                      [3, 4],
                      [5, 6]])
        Y = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0]])
        W = batch_grad_descent(X=X, Y=Y)
        np.testing.assert_equal(W.shape[0], 2)
        np.testing.assert_equal(W.shape[1], 4)


if __name__ == '__main__':
    unittest.main()
