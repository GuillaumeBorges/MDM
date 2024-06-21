from model import build_model
import numpy as np

def train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
    model = build_model((X_train.shape[1], 1))
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), verbose=1)
    return model, history
