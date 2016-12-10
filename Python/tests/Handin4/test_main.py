import unittest
import numpy as np
import numpy.testing as npt
from src.Handin4.cluster_evaluation import calculate_f1
from src.Handin4.cluster_evaluation import calcuate_contingency
from src.Handin4.cluster_evaluation import calculate_silhouettes

class EvaluationTest(unittest.TestCase):

    def test_contignency(self):
        predictions = np.array([0, 1, 1, 2])
        labels = np.array([0, 0, 1, 1])
        table = calcuate_contingency(predicted=predictions, labels=labels)
        expected = np.array((
            np.array((1, 0)),
            np.array((1, 1)),
            np.array((0, 1))
        ))
        npt.assert_array_equal(expected, table)

    def test_f1(self):
        predictions = np.array([0, 1, 1, 2])
        labels = np.array([0, 0, 1, 1])
        F_individual, F_overall, contingency = calculate_f1(predicted=predictions, labels=labels)
        self.assertAlmostEqual(11/18, F_overall, delta=0.000000001)

    def test_silhouette(self):
        data = np.array([[0,0], [3, 4], [0,0], [3, 4]])
        predictions = np.array([0, 0, 1, 1])
        silhouette_coefficient = calculate_silhouettes(data=data, predicted=predictions)
        expected = np.array([-0.5, -0.5, -0.5, -0.5])
        np.testing.assert_array_almost_equal(expected, silhouette_coefficient, decimal=8)