import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from data.data_loader import load_data_raw

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

def create_timeframe(timeframe='60min'):
   
    data: DataFrame = load_data_raw()

    # Converter a coluna de data/hora para o formato datetime
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

    # Definir a coluna de data/hora como índice
    data.set_index('Date', inplace=True)

    # Agrupar os dados em intervalos de 15 minutos
    # Aqui assumimos que você deseja manter a estrutura original.
    # Para as colunas numéricas, podemos aplicar diferentes funções de agregação
    aggregated_data = data.resample(timeframe).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
    })

    # Remover quaisquer linhas com valores NaN resultantes do agrupamento
    aggregated_data.dropna(inplace=True)

    # Resetar o índice para transformar a coluna de data/hora em uma coluna novamente
    aggregated_data.reset_index(inplace=True)

    # Filtrar os dados para incluir apenas o período entre 9:00 e 18:30
    aggregated_data = aggregated_data.set_index('Date').between_time('09:00', '18:30').reset_index()

    # Alterar o formato da data para o padrão brasileiro (dia/mês/ano)
    aggregated_data['Date'] = aggregated_data['Date'].dt.strftime('%d/%m/%Y %H:%M:%S')

    return aggregated_data

    # Salvar os dados agregados em um novo arquivo CSV
    # output_file_path = os.path.join(project_root, 'data/processed', f'winfut{timeframe}.csv')
    # aggregated_data.to_csv(output_file_path, index=False)
    # print(f"Dados agrupados salvos em: {output_file_path}")
