from django.core.management.base import BaseCommand
import os
import joblib
import numpy as np
from data_analysis.management.commands.preprocess_data import preprocess_data
from data_analysis.ml_models.algorithms.linear_regression import train_linear_regression
from data_analysis.ml_models.algorithms.random_forest import train_random_forest
from data_analysis.ml_models.algorithms.train_xgboost import train_xgboost
from data_analysis.ml_models.algorithms.arima import train_arima
from data_analysis.ml_models.algorithms.lstm import train_lstm

class Command(BaseCommand):
    help = "Train all machine learning models"

    def handle(self, *args, **kwargs):
        os.makedirs("trained_models", exist_ok=True)

        # Load preprocessed data
        train_data, _, _ = preprocess_data()
        X_train = train_data[['newCases', 'intenciveCareUnit', 'deaths']]
        y_train = train_data[['newCases', 'intenciveCareUnit', 'deaths']]

        # Train traditional ML models
        print("Training models...")
        train_linear_regression(X_train, y_train)
        train_random_forest(X_train, y_train)
        train_xgboost(X_train, y_train)

        # Train LSTM
        X_train_reshaped = np.expand_dims(X_train.values, axis=-1)
        train_lstm(X_train_reshaped, y_train)

        # Train ARIMA models per target
        for target in ['newCases', 'intenciveCareUnit', 'deaths']:
            print(f"Training ARIMA for {target}...")
            train_arima(y_train[target], target)

        print("✅ Training complete.")
