import numpy as np
from sklearn.metrics import classification_report

class ModelEvaluator:
    def create_report(self, predictions, correct_values):
        return classification_report(correct_values, predictions)

    def get_error_rate(self, predictions, correct_values):
        assert predictions.shape == correct_values.shape
        assert predictions.dtype == correct_values.dtype
        error_rate = np.mean(correct_values != predictions)
        return error_rate
