from unittest.mock import patch

from django.test import SimpleTestCase
from rest_framework.test import APIRequestFactory

from api.views import EvaluationAPIView, ModelListAPIView, load_model_object


class ModelLoadingTests(SimpleTestCase):
    @patch("api.views.load_model")
    def test_lstm_names_load_the_keras_model(self, mocked_load_model):
        expected_model = object()
        mocked_load_model.return_value = expected_model

        self.assertIs(load_model_object("lstm"), expected_model)
        self.assertIs(load_model_object("lstm_model"), expected_model)
        mocked_load_model.assert_called_with("trained_models/lstm_model.keras")

    @patch(
        "api.views.os.listdir",
        return_value=["linear_regression.pkl", "lstm_model.keras"],
    )
    def test_model_list_exposes_lstm_as_a_stable_api_name(self, _mocked_listdir):
        request = APIRequestFactory().get("/api/models/")

        response = ModelListAPIView.as_view()(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.data["available_models"],
            ["linear_regression", "lstm"],
        )


class LstmEvaluationTests(SimpleTestCase):
    @patch("api.views.evaluate_model")
    @patch("api.views.load_model_object")
    @patch("api.views.preprocess_data")
    def test_lstm_features_use_current_model_instead_of_csv(
        self, mocked_preprocess, mocked_load, mocked_evaluate
    ):
        test_data = object()
        mocked_preprocess.return_value = (object(), test_data, object())
        model = mocked_load.return_value
        mocked_evaluate.return_value = {
            feature: {"MAE": 1.0, "MSE": 2.0, "R2": 0.5}
            for feature in ("newCases", "intenciveCareUnit", "deaths")
        }

        for feature in ("newCases", "intenciveCareUnit", "deaths"):
            with self.subTest(feature=feature):
                request = APIRequestFactory().post(
                    "/api/evaluation/",
                    {"model": "lstm", "feature": feature},
                    format="json",
                )
                response = EvaluationAPIView.as_view()(request)

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.data, {
                    "model": "lstm", "feature": feature,
                    "MAE": 1.0, "MSE": 2.0, "R2": 0.5,
                })

        mocked_load.assert_called_with("lstm")
        mocked_evaluate.assert_called_with(
            model,
            test_data,
            ["newCases", "intenciveCareUnit", "deaths"],
        )

    def test_unknown_lstm_feature_returns_400(self):
        request = APIRequestFactory().post(
            "/api/evaluation/",
            {"model": "lstm", "feature": "not_a_feature"},
            format="json",
        )

        response = EvaluationAPIView.as_view()(request)

        self.assertEqual(response.status_code, 400)
