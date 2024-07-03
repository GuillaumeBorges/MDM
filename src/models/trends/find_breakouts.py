import pandas as pd


def find_breakouts(data):
    # Exemplo de propriedades baseadas em breakouts
    data['prev_high'] = data['High'].shift(1)
    data['prev_low'] = data['Low'].shift(1)

    data['breakout_up'] = (data['Close'] > data['prev_high']).astype(int)
    data['breakout_down'] = (data['Close'] < data['prev_low']).astype(int)

    # Calcular tendência baseada em breakouts
    data['uptrend'] = (data['breakout_up'].rolling(window=3).sum() > 0).astype(int)
    data['downtrend'] = (data['breakout_down'].rolling(window=3).sum() > 0).astype(int)
    data['consolidated'] = ((data['uptrend'] == 0) & (data['downtrend'] == 0)).astype(int)

    return data
