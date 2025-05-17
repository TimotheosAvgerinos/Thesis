import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for PNGs
import matplotlib.pyplot as plt



def evaluate_model(model, test_data, features, return_predictions=False, scaler=None):
    metrics = {}
    predictions = {}

    for feature in features:
        y_true_scaled = test_data[feature]
        X_test_scaled = test_data[features]

        # Predict
        if hasattr(model, 'forecast') and 'arima' in str(type(model)).lower():
            y_pred_scaled = model.forecast(steps=len(y_true_scaled))
        elif hasattr(model, 'predict'):
            if 'lstm' in str(type(model)).lower():
                X_lstm = np.expand_dims(X_test_scaled.values, axis=1)
                y_pred_all_scaled = model.predict(X_lstm)
            else:
                y_pred_all_scaled = model.predict(X_test_scaled)

            if y_pred_all_scaled.ndim == 2 and y_pred_all_scaled.shape[1] == len(features):
                idx = features.index(feature)
                y_pred_scaled = y_pred_all_scaled[:, idx]
            else:
                y_pred_scaled = y_pred_all_scaled
        else:
            continue

        # Inverse transform if scaler is provided
        if scaler:
            dummy_true = np.zeros((len(y_true_scaled), len(features)))
            dummy_pred = np.zeros((len(y_pred_scaled), len(features)))

            feature_idx = features.index(feature)
            dummy_true[:, feature_idx] = y_true_scaled
            dummy_pred[:, feature_idx] = y_pred_scaled

            y_true = scaler.inverse_transform(dummy_true)[:, feature_idx]
            y_pred = scaler.inverse_transform(dummy_pred)[:, feature_idx]
        else:
            y_true = y_true_scaled
            y_pred = y_pred_scaled

        metrics[feature] = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }

        if return_predictions:
            predictions[feature] = {
                "y_true": pd.Series(y_true).reset_index(drop=True),
                "y_pred": pd.Series(y_pred).reset_index(drop=True)
            }

    if return_predictions:
        return metrics, predictions
    return metrics


def predict_and_inverse(model, test_data, feature, features, scaler):
    y_true_scaled = test_data[feature]
    X_test_scaled = test_data[features]

    # Predict
    if hasattr(model, 'forecast') and 'arima' in str(type(model)).lower():
        y_pred_scaled = model.forecast(steps=len(y_true_scaled))
    elif hasattr(model, 'predict'):
        if 'lstm' in str(type(model)).lower():
            X_lstm = np.expand_dims(X_test_scaled.values, axis=1)
            y_pred_all = model.predict(X_lstm)
        else:
            y_pred_all = model.predict(X_test_scaled)

        if y_pred_all.ndim == 2 and y_pred_all.shape[1] == len(features):
            idx = features.index(feature)
            y_pred_scaled = y_pred_all[:, idx]
        else:
            y_pred_scaled = y_pred_all
    else:
        raise ValueError("Unsupported model type")

    # Inverse transform
    dummy_true = np.zeros((len(y_true_scaled), len(features)))
    dummy_pred = np.zeros((len(y_pred_scaled), len(features)))
    idx = features.index(feature)
    dummy_true[:, idx] = y_true_scaled
    dummy_pred[:, idx] = y_pred_scaled
    y_true = scaler.inverse_transform(dummy_true)[:, idx]
    y_pred = scaler.inverse_transform(dummy_pred)[:, idx]

    return y_true, y_pred


def plot_predictions(y_true, y_pred, model_name, feature, dates, scaler, features):
    
    # Get index of the current feature in original feature list
    feature_index = features.index(feature)

    # Rebuild full 2D arrays to apply inverse_transform
    y_true_full = np.zeros((len(y_true), len(features)))
    y_pred_full = np.zeros((len(y_pred), len(features)))
    y_true_full[:, feature_index] = y_true
    y_pred_full[:, feature_index] = y_pred

    # Apply inverse transform to return to original scale
    y_true_unscaled = scaler.inverse_transform(y_true_full)[:, feature_index]
    y_pred_unscaled = scaler.inverse_transform(y_pred_full)[:, feature_index]

    # Plot
    plt.figure(figsize=(14, 5))
    plt.plot(dates, y_true_unscaled, label='Actual', linestyle='--')
    plt.plot(dates, y_pred_unscaled, label='Predicted')
    plt.title(f'{model_name} - {feature}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend()

    # Set x-axis format for better date ticks
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_{feature}.png')
    plt.close()

