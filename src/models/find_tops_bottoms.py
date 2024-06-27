import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from data.data_loader import load_data_raw
import pandas as pd

#Carregando dados do ativo
data = load_data_raw()

# Converter a coluna de data/hora para o formato datetime
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# Definir a coluna de data/hora como índice
data.set_index('Date', inplace=True)

# Supondo que a coluna 'Close' representa os preços de fechamento
prices = data['Close']

# Identificar os picos (topos)
peaks, _ = find_peaks(prices, distance=10)  # Ajuste 'distance' conforme necessário

# Identificar os vales (fundos) invertendo os dados
inverted_prices = prices * -1
troughs, _ = find_peaks(inverted_prices, distance=10)  # Ajuste 'distance' conforme necessário

# Plotar os preços com os topos e fundos destacados
plt.figure(figsize=(14, 7))
plt.plot(prices, label='Preço de Fechamento')
plt.plot(prices.iloc[peaks], 'v', label='Topos', markersize=8, color='g')
plt.plot(prices.iloc[troughs], '^', label='Fundos', markersize=8, color='r')
plt.xlabel('Data')
plt.ylabel('Preço')
plt.title('Análise de Topos e Fundos')
plt.legend()
plt.grid(True)
plt.show()
