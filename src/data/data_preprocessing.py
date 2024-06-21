import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data: DataFrame):
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(close_prices)
    return scaled_close, scaler

def create_dataset(data: DataFrame, time_step=60):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)
