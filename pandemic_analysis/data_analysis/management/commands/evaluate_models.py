import os
import joblib
import pandas as pd
from django.core.management.base import BaseCommand
from data_analysis.management.commands.preprocess_data import preprocess_data
from data_analysis.ml_models.utils import evaluate_model
from tensorflow import keras  # for loading LSTM model

def evaluate_models():
    _, test_data, _ = preprocess_data()
    features = ['newCases', 'intenciveCareUnit', 'deaths']

    models = {}

    # Load models
    for model_name in ["linear_regression", "random_forest", "arima", "xgboost"]:
        path = f"trained_models/{model_name}.pkl"
        if os.path.exists(path):
            models[model_name] = joblib.load(path)

    # Load LSTM separately
    lstm_path = "trained_models/lstm_model.keras"
    if os.path.exists(lstm_path):
        models["lstm"] = keras.models.load_model(lstm_path)

    results = {}

    for name, model in models.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate_model(model, test_data, features)

    print("\nModel Evaluation Results:")
    for name, metrics in results.items():
        print(f"{name}: {metrics}")

    return results


class Command(BaseCommand):
    help = "Evaluate all trained machine learning models."

    def handle(self, *args, **kwargs):
        results = evaluate_models()
        self.stdout.write(self.style.SUCCESS("Model evaluation complete!"))
