import pandas as pd


def ict_trends(data):
    # Exemplo de propriedades baseadas em ICT
    data['ict_swing_high'] = data['High'].shift(1) > data['High'] & data['High'].shift(-1) > data['High']
    data['ict_swing_low'] = data['Low'].shift(1) < data['Low'] & data['Low'].shift(-1) < data['Low']

    # Calcular tendência baseada em swing highs e lows
    data['ict_uptrend'] = (data['ict_swing_low'].rolling(window=3).sum() > 0).astype(int)
    data['ict_downtrend'] = (data['ict_swing_high'].rolling(window=3).sum() > 0).astype(int)
    data['ict_consolidated'] = ((data['ict_uptrend'] == 0) & (data['ict_downtrend'] == 0)).astype(int)

    return data
