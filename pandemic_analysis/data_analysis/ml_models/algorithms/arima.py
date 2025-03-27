from statsmodels.tsa.arima.model import ARIMA
import joblib
import os

def train_arima(y_train, feature):
    order = (5, 1, 0)  # Update this if you've fine-tuned per feature
    model = ARIMA(y_train, order=order)
    model_fit = model.fit()
    
    model_path = os.path.join("trained_models", f"arima_{feature}.pkl")
    joblib.dump(model_fit, model_path)
    
    return model_fit
