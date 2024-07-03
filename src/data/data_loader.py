from typing import Any
from utils.config import load_config
import yfinance as yf
import pandas as pd
from pandas import DataFrame
from utils.helpers import read_path, read_path_dir
import os
from pyspark.sql import SparkSession


def download_data(ticker, start_date, end_date):
    data: DataFrame | Any = yf.download(ticker, start=start_date, end=end_date)
    return data


def load_data_raw() -> DataFrame:
    data: DataFrame = pd.read_csv(read_path('data/raw', 'winfut.csv'))
    return data


def get_top_100_stocks():
    # Lista das 100 ações mais negociadas.
    top_100_stocks = [
        "ELET6.SA", "B3SA3.SA", "HAPV3.SA", "COGN3.SA", "PETR4.SA",
        "BBDC4.SA", "ASAI3.SA", "VALE3.SA", "ABEV3.SA", "ITUB4.SA",
        "CVCB3.SA", "AZUL4.SA", "LREN3.SA", "MGLU3.SA", "CMIN3.SA",
        "PCAR3.SA", "ITSA4.SA", "BBAS3.SA", "BPAC11.SA", "PETR3.SA",
        "AMER3.SA", "IFCM3.SA", "GGBR4.SA", "NVDC34.SA", "CIEL3.SA",
        "CMIG4.SA", "RAIZ4.SA", "SUZB3.SA", "BBSE3.SA", "AMBP3.SA",
        "CSAN3.SA", "MRVE3.SA", "QUAL3.SA", "CCRO3.SA", "CRFB3.SA",
        "ANIM3.SA", "JBSS3.SA", "BRFS3.SA", "MRFG3.SA", "VIVT3.SA",
        "TIMS3.SA", "VAMO3.SA", "BOVA11.SA", "VVAR3.SA", "ELET3.SA",
        "RAIL3.SA", "YDUQ3.SA", "RENT3.SA", "LWSA3.SA", "BHIA3.SA",
        "GOAU4.SA", "PETZ3.SA", "VBBR3.SA", "LJQQ3.SA", "RADL3.SA",
        "BBDC3.SA", "CSNA3.SA", "USIM5.SA", "EQTL3.SA", "SIMH3.SA",
        "SULA11.SA", "PRIO3.SA", "KLBN11.SA", "MOVI3.SA", "CYRE3.SA",
        "GMAT3.SA", "CPLE6.SA", "MULT3.SA", "ECOR3.SA", "SOMA3.SA",
        "NTCO3.SA", "EMBR3.SA", "VVEO3.SA", "WEGE3.SA", "POMO4.SA",
        "ALOS3.SA", "GOLL4.SA", "AZEV4.SA", "AGXY3.SA", "TSLA34.SA",
        "AESB3.SA", "KRSA3.SA", "BRKM5.SA", "PLPL3.SA", "CEAB3.SA",
        "BOVV11.SA", "ENGI11.SA", "CBAV3.SA", "GFSA3.SA", "MLAS3.SA",
        "VIVA3.SA", "CXSE3.SA", "SULA11.SA", "CURY3.SA", "ENEV3.SA",
        "IGTI11.SA", "HYPE3.SA", "UGPA3.SA", "SBSP3.SA", "BRAP4.SA"
    ]
    return top_100_stocks


def download_and_save_stock_data(stock_symbol, save_path):
    config = load_config()

    stock_data = download_data(stock_symbol, config['start_date'], config['end_date'])
    stock_data.to_csv(save_path)


def main():
    # Inicializar SparkSession
    spark = SparkSession.builder \
        .master('local') \
        .appName("StockDataDownloader") \
        .getOrCreate()

    sc = spark.sparkContext

    top_100_stocks = get_top_100_stocks()

    # Baixar e salvar dados das ações usando Spark
    for stock in top_100_stocks:
        save_path = read_path('data/raw', f"{stock}.csv")
        sc.parallelize([stock]).foreach(lambda s: download_and_save_stock_data(s, save_path))

    spark.stop()
    print("Dados das ações salvos com sucesso!")


if __name__ == "__main__":
    main()
