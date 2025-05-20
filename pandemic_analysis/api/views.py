from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import FileResponse, HttpResponse, JsonResponse
import matplotlib.pyplot as plt
import io
import os
import numpy as np
import pandas as pd
import joblib
from django.http import FileResponse, JsonResponse
from keras.models import load_model # type: ignore
from data_analysis.management.commands.preprocess_data import preprocess_data
from data_analysis.ml_models.utils import evaluate_model, predict_and_inverse, plot_predictions
from drf_yasg.utils import swagger_auto_schema
from .serializers import PredictionRequestSerializer, FeatureListSerializer, DateRangeSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status



class ModelListAPIView(APIView):
    @swagger_auto_schema(tags=["1. Models"])
    def get(self, request):
        model_files = os.listdir("trained_models")
        models = set()

        for file in model_files:
            name, ext = os.path.splitext(file)
            if name.startswith("arima"):
                models.add("arima")
            else:
                models.add(name)

        return Response({"available_models": sorted(models)}, status=status.HTTP_200_OK)
    
class PredictAPIView(APIView):
    @swagger_auto_schema(request_body=PredictionRequestSerializer,tags=["4. Predict"])
    def post(self, request):
        serializer = PredictionRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        model_name = request.data.get("model")
        feature = request.data.get("feature")

        if not model_name or not feature:
            return Response({"error": "Both 'model' and 'feature' must be provided."}, status=400)
        
        _, test_data, scaler = preprocess_data()
        features = ['newCases', 'intenciveCareUnit', 'deaths']

        try:
            if model_name == "lstm":
                model = load_model(f"trained_models/lstm_model.keras")
            else:
                model = joblib.load(f"trained_models/{model_name}.pkl")
        except Exception as e:
            return Response({"error": f"Model loading failed: {str(e)}"}, status=500)
        
        try:
            _, preds = evaluate_model(model, test_data, features, return_predictions=True, scaler=scaler)
            pred_data = preds[feature]

            return Response({
                "dates": test_data['date'].astype(str).tolist(),
                "y_true": pred_data["y_true"].tolist(),
                "y_pred": pred_data["y_pred"].tolist()
            })
        except Exception as e:
            return Response({"error": f"Prediction failed: {str(e)}"}, status=500)
        

class EvaluationAPIView(APIView):
    @swagger_auto_schema(request_body=PredictionRequestSerializer, tags=["5. Evaluation"])
    def post(self, request):
        serializer = PredictionRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)        
        model = request.data.get('model')
        feature = request.data.get('feature')

        if not model or not feature:
            return Response(
                {"error": "Please provide 'model' and 'feature' in the request body."},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not os.path.exists('model_evaluation_results.csv'):
            return Response(
                {"error": "Evaluation results file not found."},
                status=status.HTTP_404_NOT_FOUND
            )

        df = pd.read_csv('model_evaluation_results.csv')
        row = df[(df['Model'] == model) & (df['Feature'] == feature)]

        if row.empty:
            return Response(
                {"error": f"No evaluation found for model '{model}' and feature '{feature}'."},
                status=status.HTTP_404_NOT_FOUND
            )

        result = row.iloc[0]
        return Response({
            "model": result['Model'],
            "feature": result['Feature'],
            "MAE": result['MAE'],
            "MSE": result['MSE'],
            "R2": result['R2'],
        })



def get_test_data_and_scaler():
    _, test_data, scaler = preprocess_data()
    features = ['newCases', 'intenciveCareUnit', 'deaths']
    return test_data, scaler, features

def load_model_object(model_name):
    if model_name == "lstm":
        model_path = f"trained_models/{model_name}.keras"
        return load_model(model_path)
    else:
        model_path = f"trained_models/{model_name}.pkl"
        return joblib.load(model_path)


class PlotAPIView(APIView):
    @swagger_auto_schema(request_body=PredictionRequestSerializer, tags=["6. Plot"])
    def post(self, request):
        serializer = PredictionRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        model_name = request.data.get("model")
        feature = request.data.get("feature")

        if not model_name or not feature:
            return Response({"error": "Missing model or feature"}, status=status.HTTP_400_BAD_REQUEST)

        # Load data
        _, test_data, scaler = preprocess_data()
        features = ['newCases', 'intenciveCareUnit', 'deaths']
        dates = test_data["date"].values

        # Load model
        if model_name == "lstm_model":
            model_path = os.path.join("trained_models", f"{model_name}.keras")
            model = load_model(model_path)
        else:
            model_path = os.path.join("trained_models", f"{model_name}.pkl")
            model = joblib.load(model_path)

        # Get predictions
        _, preds = evaluate_model(model, test_data, features, return_predictions=True, scaler=scaler)
        y_true = preds[feature]["y_true"]
        y_pred = preds[feature]["y_pred"]

        # Generate plot
        plot_predictions(
            y_true=y_true,
            y_pred=y_pred,
            model_name=model_name,
            feature=feature,
            dates=dates,
            scaler=scaler,
            features=features
        )

        # Return the plot as a file
        plot_path = os.path.join("plots", f"{model_name}_{feature}.png")
        if os.path.exists(plot_path):
            return FileResponse(open(plot_path, "rb"), content_type="image/png")
        else:
            return JsonResponse({"error": "Plot not found"}, status=404)
        
class FeatureListAPIView(APIView):
    @swagger_auto_schema(responses={200: FeatureListSerializer}, tags=["2. Features"])
    def get(self, request):
        features = ['newCases', 'intenciveCareUnit', 'deaths']
        return Response({"features": features})


class DateRangeAPIView(APIView):
    @swagger_auto_schema(responses={200: DateRangeSerializer}, tags=["3. Dates"])
    def get(self, request):
        _, test_data, _ = preprocess_data()
        start_date = test_data["date"].min().date()
        end_date = test_data["date"].max().date()
        return Response({"start_date": start_date, "end_date": end_date})





