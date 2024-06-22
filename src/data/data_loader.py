from typing import Any
from utils.config import load_config
import yfinance as yf
from pandas import DataFrame


def download_data(ticker, start_date, end_date):
    data: DataFrame | Any = yf.download(ticker, start=start_date, end=end_date)
    return data


config = load_config()

# Acessar as configurações
ticker = config['ticker']
start_date = config['start_date']
end_date = config['end_date']
time_step = config['time_step']
epochs = config['epochs']
batch_size = config['batch_size']

df = download_data(ticker, start_date, end_date)
print(df.head())