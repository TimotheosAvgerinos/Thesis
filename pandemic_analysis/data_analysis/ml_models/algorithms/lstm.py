import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from tensorflow.python.keras.layers import Dense,LSTM
import numpy as np
import os

def train_lstm(X_train, y_train):
    X_train = np.expand_dims(X_train, axis=1)
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(1, X_train.shape[2])),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    model.save(os.path.join("trained_models", "lstm_model.keras"))
    return model 


