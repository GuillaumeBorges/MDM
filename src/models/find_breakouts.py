import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from data.data_loader import load_data_raw


def identify_peaks_troughs(data, column='Close', distance=10):
    # Identificar os picos (topos)
    peaks, _ = find_peaks(data[column], distance=distance)

    # Identificar os vales (fundos) invertendo os dados
    inverted_prices = data[column] * -1
    troughs, _ = find_peaks(inverted_prices, distance=distance)

    return peaks, troughs


def detect_breakouts(data, peaks, troughs, column='Close'):
    breakouts = []
    for i in range(1, len(data)):
        price = data[column].iloc[i]
        if i in peaks:
            last_peak = data[column].iloc[peaks[peaks < i][-1]] if len(peaks[peaks < i]) > 0 else np.nan
            if price > last_peak:
                breakouts.append((data.index[i], 'Topo Rompido', price))
        elif i in troughs:
            last_trough = data[column].iloc[troughs[troughs < i][-1]] if len(troughs[troughs < i]) > 0 else np.nan
            if price < last_trough:
                breakouts.append((data.index[i], 'Fundo Rompido', price))
    return breakouts


def plot_with_breakouts(data, peaks, troughs, breakouts, column='Close'):
    plt.figure(figsize=(14, 7))
    plt.plot(data[column], label='Preço de Fechamento')
    plt.plot(data[column].iloc[peaks], 'v', label='Topos', markersize=8, color='g')
    plt.plot(data[column].iloc[troughs], '^', label='Fundos', markersize=8, color='r')

    for breakout in breakouts:
        date, breakout_type, price = breakout
        color = 'g' if 'Topo' in breakout_type else 'r'
        plt.plot(date, price, 'o', markersize=10, color=color, label=breakout_type)

    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.title('Análise de Topos e Fundos com Rompimentos')
    plt.legend()
    plt.grid(True)
    plt.show()


# Carregar os dados do arquivo CSV
data = load_data_raw()

# Converter a coluna de data/hora para o formato datetime
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# Definir a coluna de data/hora como índice
data.set_index('Date', inplace=True)

# Identificar os topos e fundos
peaks, troughs = identify_peaks_troughs(data)

# Detectar rompimentos
breakouts = detect_breakouts(data, peaks, troughs)

# Plotar os dados com topos, fundos e rompimentos destacados
plot_with_breakouts(data, peaks, troughs, breakouts)
