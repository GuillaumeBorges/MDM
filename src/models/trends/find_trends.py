import pandas as pd


def find_trends(data):
    # Calcular médias móveis
    data['ma50'] = data['Close'].rolling(window=50).mean()
    data['ma200'] = data['Close'].rolling(window=200).mean()

    # Calcular tendência baseada em médias móveis
    data['uptrend'] = (data['ma50'] > data['ma200']).astype(int)
    data['downtrend'] = (data['ma50'] < data['ma200']).astype(int)
    data['consolidated'] = ((data['ma50'] == data['ma200'])).astype(int)

    return data
