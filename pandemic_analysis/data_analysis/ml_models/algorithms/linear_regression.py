from sklearn.linear_model import LinearRegression
import joblib
import os

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join("trained_models", "linear_regression.pkl"))
    return model
