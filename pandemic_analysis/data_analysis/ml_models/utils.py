from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, test_data, features):
    metrics = {}

    for feature in features:
        y_true = test_data[feature]
        X_test = test_data[features]

        if hasattr(model, 'forecast') and 'arima' in str(type(model)).lower():
            y_pred = model.forecast(steps=len(y_true))

        elif hasattr(model, 'predict'):
            if 'sequential' in str(type(model)).lower():
                X_lstm = np.expand_dims(X_test.values, axis=1).astype(np.float32)
                y_pred_all = model.predict(X_lstm)
            else:
                y_pred_all = model.predict(X_test)

            if y_pred_all.ndim == 2 and y_pred_all.shape[1] == len(features):
                idx = features.index(feature)
                y_pred = y_pred_all[:, idx]
            else:
                y_pred = y_pred_all

        else:
            continue

        metrics[feature] = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }

    return metrics
