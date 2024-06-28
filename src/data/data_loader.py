from typing import Any
from utils.config import load_config
import yfinance as yf
import pandas as pd
from pandas import DataFrame
from utils.helpers import read_path


def download_data(ticker, start_date, end_date):
    data: DataFrame | Any = yf.download(ticker, start=start_date, end=end_date)
    return data


def load_data_raw() -> DataFrame:
    data: DataFrame = pd.read_csv(read_path('data/raw', 'winfut.csv'))
    return data

