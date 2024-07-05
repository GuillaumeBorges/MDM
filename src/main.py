import os
from datetime import datetime

import pandas as pd
from pyspark.sql.types import StructType, StructField, FloatType

from src.data.spark_data_processing import initialize_spark, load_data, preprocess_data, load_data_raw
from src.models.spark_model_training import train_lstm_model, evaluate_model
from src.models.trends.find_trends import find_trends
from src.models.trends.find_breakouts import find_breakouts
from src.utils.helpers import read_path_dir

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from src.models.trends.find_breakouts import find_breakouts
from src.models.trends.find_trends import find_trends
from src.models.trends.ict_trends import ict_trends


# Função para carregar e preparar os dados
def load_and_prepare_data(spark, file_path):
    data = load_data(spark, file_path)

    # Converter para Pandas DataFrame para aplicar as funções de rompimentos e tendências
    data_pd = data.toPandas()
    data_pd = find_breakouts(data_pd)
    data_pd = find_trends(data_pd)
    data_pd = ict_trends(data_pd)

    # Converter de volta para Spark DataFrame
    data = spark.createDataFrame(data_pd)

    # Selecionar apenas as colunas numéricas para o scaler
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    data_pd = data.select(numeric_columns).toPandas()

    # Normalizar os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_pd[numeric_columns] = scaler.fit_transform(data_pd[numeric_columns])

    # Unir as colunas normalizadas com o DataFrame original
    data_normalized = data_pd.join(data.drop(*numeric_columns).toPandas())

    return spark.createDataFrame(data_normalized), scaler


def create_sequences(data, seq_length, spark):
    X, y = [], []

    # Converter Spark DataFrame para Pandas DataFrame
    data_pd = data.toPandas()

    # Remover a coluna 'Date' e garantir que todas as colunas sejam numéricas
    if 'Date' in data_pd.columns:
        data_pd = data_pd.drop(columns=['Date'])

    for i in range(len(data_pd) - seq_length):
        X.append(data_pd.iloc[i:i + seq_length].values)
        y.append(data_pd.iloc[i + seq_length].values)

        # Converter listas para numpy arrays
    sequences_np = np.array(X)
    labels_np = np.array(y)

    # Verificar se todos os elementos em sequences_np são numéricos
    sequences_np = sequences_np.astype(float)
    labels_np = labels_np.astype(float)

    # Converter arrays numpy para listas de tuplas com valores float
    flattened_sequences = [tuple(map(float, seq.flatten())) for seq in sequences_np]
    flattened_labels = [tuple(map(float, label)) for label in labels_np]

    # Ajustar esquema para o DataFrame Spark
    num_features = sequences_np.shape[2]
    seq_fields = [StructField(f"seq_{i}", FloatType(), True) for i in range(seq_length * num_features)]
    schema_seq = StructType(seq_fields)
    label_fields = [StructField(f"label_{i}", FloatType(), True) for i in range(labels_np.shape[1])]
    schema_label = StructType(label_fields)

    # Criar DataFrames Spark
    sequences_spark = spark.createDataFrame(flattened_sequences, schema=schema_seq)
    labels_spark = spark.createDataFrame(flattened_labels, schema=schema_label)

    return sequences_spark, labels_spark


def main():
    spark = initialize_spark("StockAnalysis")
    data_dir = read_path_dir("data/raw")
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    seq_length = 60

    model = Sequential()

    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        df, scaler = load_and_prepare_data(spark, file_path)
        df = preprocess_data(df)

        # Criar sequências de dados
        X, y = create_sequences(df, seq_length, spark)

        # Converter DataFrames Spark para Pandas
        X_pd = X.toPandas()
        y_pd = y.toPandas()

        # Dividir os dados em conjuntos de treinamento e teste
        split = int(0.8 * len(X_pd))
        X_train, X_test = X_pd[:split], X_pd[split:]
        y_train, y_test = y_pd[:split], y_pd[split:]

        # Criar e compilar o modelo LSTM
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=3, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Treinar o modelo
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

        # Fazer previsões
        predictions = model.predict(X_test)

        # Mostrar resultados
        print(predictions)

        # Treinar o modelo LSTM com os novos indicadores
        model = train_lstm_model(df)
        predictions = evaluate_model(model, df, spark)

        # Salvar ou visualizar as previsões
        output_path = os.path.join("data/processed", f"predictions_{file_name}")
        predictions.toPandas().to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    #data = load_data_raw(spark, f"{data_dir}/data/raw/winfut.csv")
    #predictions_final = evaluate_model(model, data, spark)
    #output_path = os.path.join("data/processed", "predictions_win.csv")
    #predictions_final.toPandas().to_csv(output_path, index=False)
    #print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
