import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from data_analysis.ml_models.utils import evaluate_model


class FakeSequenceModel:
    input_shape = (None, 1, 3)

    def __init__(self):
        self.received_shape = None

    def predict(self, values, verbose=0):
        self.received_shape = values.shape
        return values[:, 0, :]


class EvaluateSequenceModelTests(SimpleTestCase):
    def test_sequence_model_is_reshaped_and_each_feature_output_is_used(self):
        features = ["newCases", "intenciveCareUnit", "deaths"]
        test_data = pd.DataFrame(
            {
                "newCases": [1.0, 2.0],
                "intenciveCareUnit": [3.0, 4.0],
                "deaths": [5.0, 6.0],
            }
        )
        model = FakeSequenceModel()

        metrics, predictions = evaluate_model(
            model,
            test_data,
            features,
            return_predictions=True,
        )

        self.assertEqual(model.received_shape, (2, 1, 3))
        for feature in features:
            self.assertEqual(metrics[feature]["MAE"], 0.0)
            np.testing.assert_array_equal(
                predictions[feature]["y_pred"].to_numpy(),
                test_data[feature].to_numpy(),
            )
