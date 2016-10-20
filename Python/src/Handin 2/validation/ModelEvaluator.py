import numpy as np

class ModelEvaluator:
    def evaluate_logistic(self, predictions, correct_values):
        assert predictions.shape == correct_values.shape
        assert predictions.dtype == correct_values.dtype
        error_rate = np.mean(correct_values != predictions)
        return error_rate
