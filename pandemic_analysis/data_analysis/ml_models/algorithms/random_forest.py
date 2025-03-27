from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join("trained_models", "random_forest.pkl"))
    return model
