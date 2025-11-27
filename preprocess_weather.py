from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# -------------------------------------------------------
# 1. Create Spark session
# -------------------------------------------------------
spark = SparkSession.builder \
    .appName("WeatherDataCleaning") \
    .getOrCreate()

# -------------------------------------------------------
# 2. Load CSV
# -------------------------------------------------------
# Update this path EXACTLY to match your laptop username
input_path = "/Users/nidhi/Downloads/boston-weather.csv"

df = spark.read.csv(input_path, header=True, inferSchema=True)

print("Raw Schema:")
df.printSchema()

# -------------------------------------------------------
# 3. Standardize column names
# -------------------------------------------------------
df = df.toDF(*[c.lower().replace(" ", "_") for c in df.columns])

# -------------------------------------------------------
# 4. Handle missing values
# -------------------------------------------------------

# Drop rows where date is missing (critical field)
if "date" in df.columns:
    df = df.dropna(subset=["date"])

# Fill numeric columns with mean
numeric_cols = [c for c, t in df.dtypes if t in ("double", "int", "float")]

for col_name in numeric_cols:
    mean_val = df.select(mean(col(col_name))).first()[0]
    if mean_val is not None:
        df = df.na.fill({col_name: mean_val})

# -------------------------------------------------------
# 5. Convert date column to proper DateType
# -------------------------------------------------------
if "date" in df.columns:
    df = df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
    df = df.filter(col("date").isNotNull())

# -------------------------------------------------------
# 6. Remove duplicates
# -------------------------------------------------------
df = df.dropDuplicates()

# -------------------------------------------------------
# 7. Detect & fix outliers (temperature example)
# -------------------------------------------------------
def cap_outliers(df, col_name):
    q1, q3 = df.approxQuantile(col_name, [0.25, 0.75], 0.05)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return df.withColumn(
        col_name,
        when(col(col_name) < lower, lower)
        .when(col(col_name) > upper, upper)
        .otherwise(col(col_name))
    )

if "temperature" in df.columns:
    df = cap_outliers(df, "temperature")

# -------------------------------------------------------
# 8. Add new useful features
# -------------------------------------------------------
if "date" in df.columns:
    df = df.withColumn("month", month(col("date"))) \
           .withColumn("year", year(col("date"))) \
           .withColumn("day_of_week", date_format(col("date"), "E"))

# -------------------------------------------------------
# 9. Save clean dataset as CSV
# -------------------------------------------------------
output_folder = "/Users/nidhi/Downloads/clean-boston-weather"

df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_folder)

print("\n✔ Clean CSV saved to:", output_folder)
print("Inside this folder you'll find one CSV file named part-00000-*.csv")

# -------------------------------------------------------
# Show sample
# -------------------------------------------------------
df.show(10)

spark.stop()




