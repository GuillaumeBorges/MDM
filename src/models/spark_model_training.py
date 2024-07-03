from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pyspark.ml.evaluation import RegressionEvaluator


def create_spark_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_lstm_model(df):
    # Conversão do DataFrame Spark para Pandas
    data_pd = df.toPandas()

    # Preparação dos dados para LSTM
    data_pd = data_pd.sort_values('Date')
    X = []
    y = []
    window_size = 60

    for i in range(window_size, len(data_pd)):
        X.append(data_pd.iloc[i - window_size:i, 1:5].values)  # Inclui Close, ICT_Trends, Trends, Breakouts
        y.append(data_pd.iloc[i, 1])

    X = np.array(X)
    y = np.array(y)

    X_train = np.reshape(X, (X.shape[0], X.shape[1], 4))  # Atualiza para incluir 4 características

    # Criação e treinamento do modelo LSTM
    model = create_spark_lstm_model((X_train.shape[1], 4))
    model.fit(X_train, y, epochs=10, batch_size=32)

    return model


def evaluate_model(model, df, spark):
    # Avaliação do modelo com dados Spark
    data_pd = df.toPandas()
    X_test = []
    window_size = 60

    for i in range(window_size, len(data_pd)):
        X_test.append(data_pd.iloc[i - window_size:i, 1:5].values)  # Inclui Close, ICT_Trends, Trends, Breakouts

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))

    predictions = model.predict(X_test)

    # Conversão de previsões para DataFrame Spark
    predictions_df = pd.DataFrame(predictions, columns=["Prediction"])
    predictions_spark_df = spark.createDataFrame(predictions_df)

    return predictions_spark_df
