from typing import Any

import yfinance as yf
from pandas import DataFrame


def download_data(ticker, start_date, end_date):
    data: DataFrame | Any = yf.download(ticker, start=start_date, end=end_date)
    return data
