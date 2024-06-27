import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from data.data_loader import load_data_raw


def calculate_moving_averages(data, window_short=50, window_long=200):
    data['SMA50'] = data['Close'].rolling(window=window_short).mean()
    data['SMA200'] = data['Close'].rolling(window=window_long).mean()
    return data


def identify_peaks_troughs(data, column='Close', distance=10):
    peaks, _ = find_peaks(data[column], distance=distance)
    inverted_prices = data[column] * -1
    troughs, _ = find_peaks(inverted_prices, distance=distance)
    return peaks, troughs


def detect_breakouts(data, peaks, troughs, column='Close'):
    data['TopBreakout'] = 0
    data['BottomBreakout'] = 0
    for i in range(1, len(data)):
        price = data[column].iloc[i]
        if i in peaks:
            last_peak = data[column].iloc[peaks[peaks < i][-1]] if len(peaks[peaks < i]) > 0 else np.nan
            if price > last_peak:
                data.at[data.index[i], 'TopBreakout'] = 1
        elif i in troughs:
            last_trough = data[column].iloc[troughs[troughs < i][-1]] if len(troughs[troughs < i]) > 0 else np.nan
            if price < last_trough:
                data.at[data.index[i], 'BottomBreakout'] = 1
    return data


def determine_trend(data):
    data['Trend'] = np.where(data['Close'] > data['SMA200'], 1, -1)
    return data


def create_dataset(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), :]
        X.append(a)
        y.append(data[i + look_back, -1])  # Última coluna é a tendência
    return np.array(X), np.array(y)


# Carregar os dados do arquivo CSV
data = load_data_raw()

# Converter a coluna de data/hora para o formato datetime
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# Definir a coluna de data/hora como índice
data.set_index('Date', inplace=True)

# Calcular as médias móveis
data = calculate_moving_averages(data)

# Identificar os topos e fundos
peaks, troughs = identify_peaks_troughs(data)

# Detectar rompimentos
data = detect_breakouts(data, peaks, troughs)

# Determinar a tendência
data = determine_trend(data)

# Remover NaNs criados pelas médias móveis
data.dropna(inplace=True)

# Normalizar os dados
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close', 'SMA50', 'SMA200', 'TopBreakout', 'BottomBreakout', 'Trend']])

# Criar o conjunto de dados para a LSTM
look_back = 10  # Número de passos no tempo para olhar para trás
X, y = create_dataset(scaled_data, look_back)

# Redimensionar os dados para [amostras, passos de tempo, características]
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

# Construir a Rede Neural LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, X.shape[2])))
model.add(LSTM(50))
model.add(Dense(1, activation='tanh'))  # A ativação 'tanh' é usada para saída entre -1 e 1

# Compilar o modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Treinar o modelo
model.fit(X, y, epochs=50, batch_size=32, verbose=1)

# Fazer previsões
predicted = model.predict(X)

# Plotar os resultados
plt.figure(figsize=(14, 7))
plt.plot(data.index[look_back + 1:], data['Trend'][look_back + 1:], label='Tendência Real')
plt.plot(data.index[look_back + 1:], predicted, label='Tendência Prevista')
plt.xlabel('Data')
plt.ylabel('Tendência')
plt.title('Previsão de Tendência com LSTM')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
