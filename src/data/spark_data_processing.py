from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date


def initialize_spark(app_name="StockAnalysis"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark


def load_data(spark, file_path):
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    return df


def preprocess_data(df):
    df = df.withColumn("Date", to_date(col("Date"), "dd/MM/yyyy HH:mm"))
    # Outras etapas de pré-processamento podem ser adicionadas aqui
    return df
