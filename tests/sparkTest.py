from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql import SparkSession

spark = (
    SparkSession
    .builder
    .master("local[*]")
    .appName("test")
    .getOrCreate()
)
