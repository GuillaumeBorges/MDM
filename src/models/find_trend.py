import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data.data_loader import load_data_raw

def calculate_moving_averages(data, window_short=50, window_long=200):
    data['SMA50'] = data['Close'].rolling(window=window_short).mean()
    data['SMA200'] = data['Close'].rolling(window=window_long).mean()
    return data

def determine_trend(data):
    if data['SMA50'].iloc[-1] > data['SMA200'].iloc[-1]:
        trend = 'Alta'
    else:
        trend = 'Baixa'
    return trend

def plot_trend(data, trend):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Preço de Fechamento', linewidth=1)
    plt.plot(data['SMA50'], label='SMA50', linewidth=1)
    plt.plot(data['SMA200'], label='SMA200', linewidth=1)
    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.title(f'Tendência Atual: {trend}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Carregar os dados do arquivo CSV
data = load_data_raw()

# Converter a coluna de data/hora para o formato datetime
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# Definir a coluna de data/hora como índice
data.set_index('Date', inplace=True)

# Calcular as médias móveis
data = calculate_moving_averages(data)

# Determinar a tendência atual
trend = determine_trend(data)

# Plotar os dados com as médias móveis e a tendência
plot_trend(data, trend)

print(f"Tendência Atual: {trend}")
