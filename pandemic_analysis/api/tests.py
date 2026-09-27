from unittest.mock import patch

from django.test import SimpleTestCase
from rest_framework.test import APIRequestFactory

from api.views import (
    EvaluationAPIView,
    ModelListAPIView,
    PlotAPIView,
    load_model_object,
)


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


class PlotTests(SimpleTestCase):
    @patch("api.views.os.path.exists", return_value=False)
    @patch("api.views.plot_predictions")
    @patch("api.views.evaluate_model")
    @patch("api.views.load_model_object")
    @patch("api.views.preprocess_data")
    def test_plot_predictions_are_only_inverse_scaled_once(
        self,
        mocked_preprocess,
        mocked_load,
        mocked_evaluate,
        mocked_plot,
        _mocked_exists,
    ):
        class DateColumn:
            values = ["2022-01-22"]

        class TestData:
            def __getitem__(self, key):
                if key == "date":
                    return DateColumn()
                raise KeyError(key)

        test_data = TestData()
        scaler = object()
        model = mocked_load.return_value
        mocked_preprocess.return_value = (object(), test_data, scaler)
        mocked_evaluate.return_value = ({}, {
            "newCases": {"y_true": [0.1], "y_pred": [0.2]},
        })

        request = APIRequestFactory().post(
            "/api/plot/",
            {"model": "linear_regression", "feature": "newCases"},
            format="json",
        )
        response = PlotAPIView.as_view()(request)

        self.assertEqual(response.status_code, 404)
        mocked_evaluate.assert_called_once_with(
            model,
            test_data,
            ["newCases", "intenciveCareUnit", "deaths"],
            return_predictions=True,
        )
        mocked_plot.assert_called_once_with(
            y_true=[0.1],
            y_pred=[0.2],
            model_name="linear_regression",
            feature="newCases",
            dates=["2022-01-22"],
            scaler=scaler,
            features=["newCases", "intenciveCareUnit", "deaths"],
        )
