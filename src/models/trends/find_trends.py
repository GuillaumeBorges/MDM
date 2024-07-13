import pandas as pd


def find_trends(data):
    # Calcular médias móveis
    data['ma50'] = data['Close'].rolling(window=50).mean()
    data['ma200'] = data['Close'].rolling(window=200).mean()

    #df = pd.DataFrame()

    # Calcular tendência baseada em médias móveis
    data['media_uptrend'] = (data['ma50'] > data['ma200']).astype(int)
    data['media_downtrend'] = (data['ma50'] < data['ma200']).astype(int)
    data['media_consolidated'] = ((data['ma50'] == data['ma200'])).astype(int)

    return data
