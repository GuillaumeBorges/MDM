import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data.data_loader import load_data_raw

# Carregar os dados do arquivo CSV
data = load_data_raw()

# Converter a coluna de data/hora para o formato datetime
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data.set_index('Date', inplace=True)


# Função para identificar mudanças na estrutura de mercado
def identify_market_structure(data):
    data['HigherHigh'] = data['High'] > data['High'].shift(1)
    data['HigherLow'] = data['Low'] > data['Low'].shift(1)
    data['LowerHigh'] = data['High'] < data['High'].shift(1)
    data['LowerLow'] = data['Low'] < data['Low'].shift(1)

    data['BullishStructure'] = (data['HigherHigh'] & data['HigherLow'])
    data['BearishStructure'] = (data['LowerHigh'] & data['LowerLow'])

    return data


# Função para identificar blocos de ordens
def identify_order_blocks(data, lookback=10):
    data['OrderBlockHigh'] = data['High'].rolling(window=lookback).max()
    data['OrderBlockLow'] = data['Low'].rolling(window=lookback).min()
    return data


# Aplicar as funções
data = identify_market_structure(data)
data = identify_order_blocks(data)

# Plotar os resultados
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Preço de Fechamento')
plt.plot(data.index, data['OrderBlockHigh'], label='Order Block High', linestyle='--')
plt.plot(data.index, data['OrderBlockLow'], label='Order Block Low', linestyle='--')
plt.fill_between(data.index, data['OrderBlockLow'], data['OrderBlockHigh'], color='gray', alpha=0.3)

# Marcar estruturas de mercado bullish e bearish
bullish_dates = data[data['BullishStructure']].index
bearish_dates = data[data['BearishStructure']].index
plt.scatter(bullish_dates, data.loc[bullish_dates]['Close'], marker='^', color='g', label='Bullish Structure', s=100)
plt.scatter(bearish_dates, data.loc[bearish_dates]['Close'], marker='v', color='r', label='Bearish Structure', s=100)

plt.xlabel('Data')
plt.ylabel('Preço')
plt.title('Identificação de Blocos de Ordens e Estruturas de Mercado')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
