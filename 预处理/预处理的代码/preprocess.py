from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, to_date
from pyspark.sql.types import IntegerType, FloatType

spark = SparkSession.builder.appName("SpotifyPreprocessing").master("local[*]").config("spark.sql.adaptive.enabled", "true").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("Spotify Data Preprocessing")
print("=" * 60)

df = spark.read.option("header", "true").option("inferSchema", "true").csv("hdfs://localhost:9000/user/spotify/input/spotify_2024.csv")
print(f"Original rows: {df.count():,}")

df_clean = df.withColumnRenamed("Source.Name", "source_file").withColumnRenamed("artist_names", "artist_name").withColumn("chart_date", to_date(regexp_extract(col("source_file"), r"(\d{4}-\d{2}-\d{2})", 1), "yyyy-MM-dd")).withColumn("streams", col("streams").cast(FloatType())).filter(col("streams") > 0).filter(col("streams") != -1).filter(col("chart_date").isNotNull())

print(f"Cleaned rows: {df_clean.count():,}")
df_clean.show(5, truncate=False)

df_clean.write.mode("overwrite").parquet("hdfs://localhost:9000/user/spotify/spotify_clean.parquet")
print("Preprocessing completed successfully!")
spark.stop()
