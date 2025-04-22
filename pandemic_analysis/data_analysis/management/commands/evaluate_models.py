import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from django.core.management.base import BaseCommand
from data_analysis.management.commands.preprocess_data import preprocess_data
from data_analysis.ml_models.utils import evaluate_model, plot_predictions
from keras import models 
from keras.models import load_model # type: ignore




def evaluate_models():
    _, test_data, scaler = preprocess_data()
    features = ['newCases', 'intenciveCareUnit', 'deaths']
    
    models = {}
    for model_name in ["linear_regression", "random_forest", "arima", "lstm", "xgboost"]:
        if model_name == "lstm":
            model_path = f"trained_models/{model_name}.keras"
            if os.path.exists(model_path):
                models[model_name] = load_model(model_path)
        else:
            model_path = f"trained_models/{model_name}.pkl"
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)

    
    results = {}
    all_predictions = []

    for name, model in models.items():
        print(f"Evaluating {name}...")
        metrics, preds = evaluate_model(model, test_data, features, return_predictions=True)
        results[name] = metrics
        for feature, data in preds.items():
            all_predictions.append({
                "model": name,
                "feature": feature,
                "y_true": data["y_true"],
                "y_pred": data["y_pred"]
            })
    
    # Save results to CSV
    records = []
    for model_name, metric in results.items():
        for feature, values in metric.items():
            records.append({
                "Model": model_name,
                "Feature": feature,
                "MAE": values['MAE'],
                "MSE": values['MSE'],
                "R2": values['R2']
            })
    pd.DataFrame(records).to_csv("model_evaluation_results.csv", index=False)

    # Plot predictions
    for entry in all_predictions:
        plot_predictions(entry["y_true"],
                         entry["y_pred"],
                         entry["model"],
                         entry["feature"],
                         test_data["date"].values,
                         scaler,
                         features)
    
    print("\nModel evaluation complete! Results saved to 'model_evaluation_results.csv' and plots generated.")
    return results

class Command(BaseCommand):
    help = "Evaluate machine learning models on pandemic data."

    def handle(self, *args, **kwargs):
        evaluate_models()
