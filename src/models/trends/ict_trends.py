import pandas as pd


def ict_trends(data):
    # Exemplo de propriedades baseadas em ICT
    data['HigherHigh'] = data['High'] > data['High'].shift(1)
    data['HigherLow'] = data['Low'] > data['Low'].shift(1)
    data['LowerHigh'] = data['High'] < data['High'].shift(1)
    data['LowerLow'] = data['Low'] < data['Low'].shift(1)

    data['BullishStructure'] = (data['HigherHigh'] & data['HigherLow'])
    data['BearishStructure'] = (data['LowerHigh'] & data['LowerLow'])

    data['OrderBlockHigh'] = data['High'].rolling(window=10).max()
    data['OrderBlockLow'] = data['Low'].rolling(window=10).min()

    # Adicionar colunas de tendência
    data['ict_uptrend'] = data['BullishStructure'].rolling(window=10).sum().gt(0).astype(int)
    data['ict_downtrend'] = data['BearishStructure'].rolling(window=10).sum().gt(0).astype(int)
    data['ict_consolidated'] = ((data['ict_uptrend'] == 0) & (data['ict_downtrend'] == 0)).astype(int)

    return data

