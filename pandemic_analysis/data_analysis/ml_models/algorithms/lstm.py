import numpy as np
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM


def train_lstm(X_train, y_train, epochs=50, batch_size=32):
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)

    if X_train.ndim == 2:
        X_train = np.expand_dims(X_train, axis=1)
    if X_train.ndim != 3:
        raise ValueError(
            f"LSTM training data must be 2D or 3D, got shape {X_train.shape}."
        )

    if y_train.ndim == 1:
        y_train = np.expand_dims(y_train, axis=1)

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, activation='relu', return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(y_train.shape[1])
    ])
    model.compile(optimizer='adam', loss='mse')

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )
    model.save(os.path.join("trained_models", "lstm_model.keras"))
    return model

