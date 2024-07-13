from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
from pyspark.sql.functions import col, regexp_replace, udf
from pyspark.sql.types import DoubleType

numeric_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def initialize_spark(app_name="StockAnalysis"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark


def load_data(spark, file_path):
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    return df


def load_data_raw(spark, file_path):
    df = load_data(spark, file_path)
    return df

def preprocess_data(df):
    df = df.withColumn("Date", to_date(col("Date"), "dd/MM/yyyy"))
    df.index = df['Date']

    # Registrar a função UDF
    convert_udf = udf(american_to_brazilian_number, DoubleType())

    # Aplicar a função UDF para converter os valores
    for col_name in numeric_columns:
        df = df.withColumn(col_name, convert_udf(col(col_name).cast("string")))

    # Outras etapas de pré-processamento podem ser adicionadas aqui
    return df


def american_to_brazilian_number(num_str):
    if num_str is not None:
        num_str = num_str.replace(",", "")
        num_str = num_str.replace(".", ",")
        return float(num_str.replace(",", "."))
    return None
