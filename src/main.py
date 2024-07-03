import os
import pandas as pd

from data.data_loader import load_data_raw
from src.data.spark_data_processing import initialize_spark, load_data, preprocess_data
from src.models.spark_model_training import train_lstm_model, evaluate_model
from src.models.trends.find_trends import find_trends
from src.models.trends.find_breakouts import find_breakouts

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
    #data = pd.read_csv(file_path, index_col='Date', parse_dates=True)

    # Aplicar as funções para encontrar rompimentos e tendências
    data = data.toPandas()
    data = find_breakouts(data)
    data = find_trends(data)
    data = ict_trends(data)
    data = spark.createDataFrame(data)

    # Normalizar os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :-3])
        y.append(data[i + seq_length, -3:])
    return np.array(X), np.array(y)


def main():
    spark = initialize_spark("StockAnalysis")
    data_dir = os.path.join("data/raw")
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    seq_length = 60

    model = Sequential()

    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        df, scaler = load_and_prepare_data(spark, file_path)
        df = preprocess_data(df)

        # Criar sequências de dados
        X, y = create_sequences(df, seq_length)

        # Calcular os indicadores de tendência
        #ict_trends = calculate_ict_trends(df)
        #trends = find_trends(df)
        #breakouts = find_breakouts(df)

        # Dividir os dados em conjuntos de treinamento e teste
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

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

        # Adicionar esses indicadores ao dataframe
        #df = df.toPandas()
        #df['ICT_Trends'] = ict_trends
        #df['Trends'] = trends
        #df['Breakouts'] = breakouts
        #df = spark.createDataFrame(df)

        # Treinar o modelo LSTM com os novos indicadores
        model = train_lstm_model(df)
        predictions = evaluate_model(model, df, spark)

        # Salvar ou visualizar as previsões
        output_path = os.path.join("data/processed", f"predictions_{file_name}")
        predictions.toPandas().to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    data = load_data_raw()
    predictions_final = evaluate_model(model, data, spark)
    output_path = os.path.join("data/processed", "predictions_win.csv")
    predictions_final.toPandas().to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
