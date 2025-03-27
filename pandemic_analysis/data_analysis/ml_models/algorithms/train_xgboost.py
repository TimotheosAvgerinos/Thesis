from xgboost import XGBRegressor
import joblib
import os

def train_xgboost(X_train, y_train):
    model = XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join("trained_models", "xgboost.pkl"))
    return model