import unittest
import numpy as np
from logisticregression import log_cost
class MyTestCase(unittest.TestCase):
    def test_single_positiv_column(self):
        X = np.matrix([[1, 1]])
        y = np.array([1])
        w = np.array([3, 4])
        cost, grad = log_cost(X, y, w, reg=0)
        np.testing.assert_almost_equal(cost, 0.00091146645377420212, 13)

    def test_single_negative_column(self):
        X = np.matrix([[1, 1]])
        y = np.array([0])
        w = np.array([3, 4])
        cost, grad = log_cost(X, y, w, reg=0)
        np.testing.assert_almost_equal(cost, 7.0009114664538208, 13)

    def test_double_column(self):
        X = np.matrix([[1, 1], [2, 2]])
        y = np.array([0, 1])
        w = np.array([3, 4])
        cost, grad = log_cost(X, y, w, reg=0)
        np.testing.assert_almost_equal(cost, 14.002734399361415, 13)
