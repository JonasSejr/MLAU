import unittest
import numpy as np
from mpmath import limit

from logisticregression import log_cost
from logisticregression import batch_grad_descent
from logisticregression import execute_LGMobel

#Remember nosetests should be executed with --nocapture to see output
class Log_CostTestCase(unittest.TestCase):
    def test_cost_single_positiv_column(self):
        X = np.array([[1, 1]])
        y = np.array([[1]])
        w = np.array([[3], [4]])
        cost, grad = log_cost(X, y, w, reg=0)
        np.testing.assert_almost_equal(cost, 0.00091146645377420212, 13)

    def test_cost_single_negative_column(self):
        X = np.array([[2, 2]])
        y = np.array([[0]])
        w = np.array([[3], [4]])
        cost, grad = log_cost(X, y, w, reg=0)
        np.testing.assert_almost_equal(cost, 14.000000831404552, 13)

    def test_cost_double_column(self):
        X = np.array([[1, 1], [2, 2]])
        y = np.array([[1], [0]])
        w = np.array([[3], [4]])
        cost, grad = log_cost(X, y, w, reg=0)
        np.testing.assert_almost_equal(cost, 7.000456148929163, 13)

    def test_gradient__with_one_dimensional_arrays(self):
        X = np.array([[1, 1], [2, 2], [3, 3]])
        y = np.array([1, 0, 1])
        w = np.array([3, 4])
        cost, grad = log_cost(X, y, w, reg=0)
        np.testing.assert_almost_equal(grad[0], 1.9990872834747755, 13)
        np.testing.assert_almost_equal(grad[0], 1.9990872834747755, 13)

    def test_gradient_column(self):
        X = np.array([[1, 1], [2, 2], [3, 3]])
        y = np.array([[1], [0], [1]])
        w = np.array([[3], [4]])
        cost, grad = log_cost(X, y, w, reg=0)
        np.testing.assert_almost_equal(grad[0], 1.9990872834747755, 13)
        np.testing.assert_almost_equal(grad[0], 1.9990872834747755, 13)
        #-(1* (1 - (1/(1 + np.exp(-7)))) + 2* (0 - (1/(1 + np.exp(-14)))) + 3* (1 - (1/(1 + np.exp(-21)))))

class Batch_grad_descentTestCase(unittest.TestCase):
    def test_4_points_on_line(self):
        X = np.array([[1, 1, 1], [1, 2, 2], [1, 3, 3], [1, 4, 4], [1, 2, 0], [1, 0, 2]])
        y = np.array([[1], [1], [0], [0], [1], [0]])
        w = batch_grad_descent(X,y)
        np.testing.assert_equal(actual=len(w.shape), desired=2)
        print(w)
        np.testing.assert_equal(actual=w.shape[0], desired=X.shape[1])
        np.testing.assert_equal(actual=w.shape[1], desired=1)

    def test_4_points_on_line_with_onedimension_y(self):
        X = np.array([[1, 1, 1], [1, 2, 2], [1, 3, 3], [1, 4, 4], [1, 2, 0], [1, 0, 2]])
        y = np.array([1, 1, 0, 0, 1, 0])
        w = batch_grad_descent(X, y)
        np.testing.assert_equal(actual=len(w.shape), desired=2)
        print(w)
        np.testing.assert_equal(actual=w.shape[0], desired=X.shape[1])
        np.testing.assert_equal(actual=w.shape[1], desired=1)

class execute_LGMobelTestCase(unittest.TestCase):
    def test_model_execution_positive(self):
        w = np.array([1, 2, 3])
        x = np.array([1, 1, 1])
        limit = 0.5
        np.testing.assert_equal(execute_LGMobel(w, x, limit), desired=1)
        #1/(1 + np.exp(-6))

    def test_model_execution_negative(self):
        w = np.array([1, 2, 3])
        x = np.array([1, -1, -1])
        limit = 0.5
        np.testing.assert_equal(execute_LGMobel(w, x, limit), desired=0)
        #1/(1 + np.exp(4))