import os
from datetime import datetime

import pandas as pd
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType

from src.data.spark_data_processing import initialize_spark, load_data, preprocess_data, load_data_raw
from src.models.spark_model_training import train_lstm_model, evaluate_model
from src.models.trends.find_trends import find_trends
from src.models.trends.find_breakouts import find_breakouts
from src.utils.helpers import read_path_dir

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputSpec
from tensorflow.keras.optimizers import Adam
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

    # Consolidar as informações de tendência em uma coluna
    data_pd['trend'] = 'consolidated'
    data_pd.loc[data_pd['uptrend'] == 1, 'trend'] = 'uptrend'
    data_pd.loc[data_pd['downtrend'] == 1, 'trend'] = 'downtrend'

    # Adicionar colunas de tendência ict
    data_pd['trend_ict'] = 'ict_consolidated'
    data_pd.loc[data_pd['ict_uptrend'] == 1, 'trend_ict'] = 'ict_uptrend'
    data_pd.loc[data_pd['ict_downtrend'] == 1, 'trend_ict'] = 'ict_downtrend'

    # Adicionar colunas de tendência das médias móveis
    data_pd['trend_media'] = 'media_consolidated'
    data_pd.loc[data_pd['media_uptrend'] == 1, 'trend_media'] = 'media_uptrend'
    data_pd.loc[data_pd['media_downtrend'] == 1, 'trend_media'] = 'media_downtrend'

    # Codificar as colunas de tendência
    trend_map = {'consolidated': 0, 'uptrend': 1, 'downtrend': 2}
    trend_media_map = {'media_consolidated': 0, 'media_uptrend': 1, 'media_downtrend': 2}
    trend_ict_map = {'ict_consolidated': 0, 'ict_uptrend': 1, 'ict_downtrend': 2}
    data_pd['trend'] = data_pd['trend'].map(trend_map)
    data_pd['trend_media'] = data_pd['trend_media'].map(trend_media_map)
    data_pd['trend_ict'] = data_pd['trend_ict'].map(trend_ict_map)

    # Converter de volta para Spark DataFrame
    data = spark.createDataFrame(data_pd)

    # Selecionar apenas as colunas numéricas para o scaler
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'trend', 'trend_ict', 'trend_media']
    # Manter apenas as colunas necessárias para a LSTM
    #data_pd = data_pd[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'trend', 'trend_ict', 'trend_media']]

    data_pd = data.select(numeric_columns).toPandas()

    # Normalizar os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_pd[numeric_columns] = scaler.fit_transform(data_pd[numeric_columns])

    # Unir as colunas normalizadas com o DataFrame original
    data_normalized = data_pd.join(data.drop(*numeric_columns).toPandas())

    #print(data_normalized.head())

    return spark.createDataFrame(data_normalized), scaler


def create_sequences(data, seq_length, spark):
    X = []
    y_trend = []
    y_trend_media = []
    y_trend_ict = []

    # Converter Spark DataFrame para Pandas DataFrame
    data_pd = data.toPandas()

    # Remover a coluna 'Date' e garantir que todas as colunas sejam numéricas
    if 'Date' in data_pd.columns:
        data_pd = data_pd.drop(columns=['Date'])

    for i in range(len(data_pd) - seq_length):
        X.append(data_pd.iloc[i:i + seq_length].values)
        y_trend.append(data_pd.iloc[i + seq_length]['trend'])
        y_trend_media.append(data_pd.iloc[i + seq_length]['trend_media'])
        y_trend_ict.append(data_pd.iloc[i + seq_length]['trend_ict'])

    # Verificar se as dimensões dos arrays estão corretas
    if len(X) == 0 or len(y_trend) == 0 or len(y_trend_media) == 0 or len(y_trend_ict) == 0:
        raise ValueError("Erro ao criar sequências: listas vazias encontradas.")

    # Converter listas para numpy arrays
    sequences_np = np.array(X)
    labels_np = np.array(y_trend)
    labels_np_media = np.array(y_trend_media)
    labels_np_ict = np.array(y_trend_ict)

    # Verificar se todos os elementos em sequences_np são numéricos
    sequences_np = sequences_np.astype(float)
    labels_np = labels_np.astype(int)
    labels_np_media = labels_np_media.astype(int)
    labels_np_ict = labels_np_ict.astype(int)

    # Converter arrays numpy para listas de tuplas com valores float
    flattened_sequences = [tuple(map(float, seq.flatten())) for seq in sequences_np]
    flattened_labels = [tuple(map(int, labels_np))]
    flattened_labels_media = [tuple(map(int, labels_np_media))]
    flattened_labels_ict = [tuple(map(int, labels_np_ict))]

    # Ajustar esquema para o DataFrame Spark
    num_features = sequences_np.shape[2]
    seq_fields = [StructField(f"seq_{i}", FloatType(), True) for i in range(seq_length * num_features)]
    schema_seq = StructType(seq_fields)
    label_fields = [StructField(f"label_{i}", IntegerType(), True) for i in range(labels_np.shape[0])]
    schema_label = StructType(label_fields)

    label_fields_media = [StructField(f"label_media_{i}", IntegerType(), True) for i in range(labels_np_media.shape[0])]
    schema_label_media = StructType(label_fields_media)

    label_fields_ict = [StructField(f"label_ict_{i}", IntegerType(), True) for i in range(labels_np_ict.shape[0])]
    schema_label_ict = StructType(label_fields_ict)

    # Criar DataFrames Spark
    sequences_spark = spark.createDataFrame(flattened_sequences, schema=schema_seq)
    labels_spark = spark.createDataFrame(flattened_labels, schema=schema_label)
    labels_spark_media = spark.createDataFrame(flattened_labels_media, schema=schema_label_media)
    labels_spark_ict = spark.createDataFrame(flattened_labels_ict, schema=schema_label_ict)

    return sequences_spark, labels_spark, labels_spark_media, labels_spark_ict


def main():
    spark = initialize_spark("StockAnalysis")
    data_dir = read_path_dir("data/raw")
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and not f.startswith('winfut')]
    seq_length = 60

    model = Sequential()

    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        df, scaler = load_and_prepare_data(spark, file_path)
        df = preprocess_data(df)

        # Criar sequências de dados
        X, y_trend, y_trend_media, y_trend_ict = create_sequences(df, seq_length, spark)



        # Converter DataFrames Spark para Pandas
        X_pd = X.toPandas()
        y_trend_pd = y_trend.toPandas()
        y_trend_media_pd = y_trend_media.toPandas()
        y_trend_ict_pd = y_trend_ict.toPandas()

        print(f"X shape: {X_pd.shape}")
        print(f"y_trend shape: {y_trend_pd.shape}")
        print(f"y_trend_media shape: {y_trend_media_pd.shape}")
        print(f"y_trend_ict shape: {y_trend_ict_pd.shape}")

        # Dividir os dados em conjuntos de treinamento e teste
        split = int(0.8 * len(X_pd))
        X_train, X_test = X_pd[:split], X_pd[split:]
        y_trend_train, y_trend_test = y_trend_pd[:split], y_trend_pd[split:]
        y_trend_media_train, y_trend_media_test = y_trend_media_pd[:split], y_trend_media_pd[split:]
        y_trend_ict_train, y_trend_ict_test = y_trend_ict_pd[:split], y_trend_ict_pd[split:]

        print(f'O Shape de X é {X_pd.shape}')
        # Criar o modelo LSTM com saídas múltiplas
        input_layer = Input(shape=(seq_length, X_pd.shape[1]))
        lstm_layer = LSTM(units=50, return_sequences=True)(input_layer)
        lstm_layer = LSTM(units=50)(lstm_layer)

        trend_output = Dense(3, activation='softmax', name='trend_output')(lstm_layer)
        trend_media_output = Dense(3, activation='softmax', name='trend_media_output')(lstm_layer)
        trend_ict_output = Dense(3, activation='softmax', name='trend_ict_output')(lstm_layer)

        model = Model(inputs=input_layer, outputs=[trend_output, trend_media_output, trend_ict_output])
        model.compile(optimizer=Adam(),
                      loss={'trend_output': 'sparse_categorical_crossentropy',
                            'trend_media_output': 'sparse_categorical_crossentropy',
                            'trend_ict_output': 'sparse_categorical_crossentropy'},
                      metrics={'trend_output': 'accuracy', 'trend_media_output': 'accuracy',
                               'trend_ict_output': 'accuracy'},
                      loss_weights={'trend_output': 0.5, 'trend_media_output': 1.0,
                                    'trend_ict_output': 2.0})  # Definir os pesos da perda

        # Treinar o modelo
        model.fit(X_train,
                  {'trend_output': y_trend_train, 'trend_media_output': y_trend_media_train,
                   'trend_ict_output': y_trend_ict_train},
                  epochs=50,
                  batch_size=32,
                  validation_data=(
                      X_test, {'trend_output': y_trend_test, 'trend_media_output': y_trend_media_test,
                               'trend_ict_output': y_trend_ict_test}))

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

        # data = load_data_raw(spark, f"{data_dir}/data/raw/winfut.csv")
        # predictions_final = evaluate_model(model, data, spark)
        # output_path = os.path.join("data/processed", "predictions_win.csv")
        # predictions_final.toPandas().to_csv(output_path, index=False)
        # print(f"Predictions saved to {output_path}")

        return model


if __name__ == "__main__":
    main()
