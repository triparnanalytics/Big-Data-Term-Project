############################################################
# Crime + Weather ML Forecasting Pipeline (GCP Dataproc)
# FINAL VERSION — Target = offense_code_group
# Simple, robust features; no missing columns.
############################################################

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from google.cloud import storage

# ---------------------------------------------------------
# 1. Spark Session
# ---------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("CrimeWeatherForecasting")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)

print("\n=== SPARK SESSION STARTED ===\n")

# ---------------------------------------------------------
# 2. File Paths
# ---------------------------------------------------------
CRIME_PATH = "gs://climate_crime_analysis/crime.csv"
WEATHER_PATH = "gs://climate_crime_analysis/boston-weather.csv"
NEIGH_PATH = "gs://climate_crime_analysis/Boston_Neighborhoods.geojson"

LOCAL_OUTPUT = "analysis_output.txt"
GCS_BUCKET = "climate_crime_analysis"
GCS_OUTPUT_PATH = "output/results.txt"  # <--- as you requested

# ---------------------------------------------------------
# 3. Load Weather Data
# ---------------------------------------------------------
weather_raw = (
    spark.read.option("header", "true")
    .option("inferSchema", "true")
    .csv(WEATHER_PATH)
)

weather_df = (
    weather_raw
    .withColumnRenamed("Day", "day_str")
    .withColumnRenamed("High(°F)", "high_f")
    .withColumnRenamed("Low(°F)", "low_f")
    .withColumnRenamed("Precip.(inch)", "precip_inch")
    .withColumnRenamed("Snow(inch)", "snow_inch")
    .withColumnRenamed("Snowdepth(inch)", "snowdepth_inch")
)

weather_df = weather_df.replace("-", None)

weather_df = weather_df.withColumn(
    "date",
    to_date(col("day_str"), "d MMM yyyy")
)

weather_df = (
    weather_df
    .withColumn("high_f", col("high_f").cast("double"))
    .withColumn("low_f", col("low_f").cast("double"))
    .withColumn("precip_inch", col("precip_inch").cast("double"))
    .withColumn("snow_inch", col("snow_inch").cast("double"))
    .withColumn("snowdepth_inch", col("snowdepth_inch").cast("double"))
)

print("Weather data cleaned.")

# ---------------------------------------------------------
# 4. Load Crime Data
# ---------------------------------------------------------
crime_raw = (
    spark.read.option("header", "true")
    .option("inferSchema", "true")
    .csv(CRIME_PATH)
)

# Convert all columns to lowercase
crime_df = crime_raw.toDF(*[c.lower() for c in crime_raw.columns])

crime_df = crime_df.withColumn(
    "occurred_on_date", to_timestamp(col("occurred_on_date"))
)

crime_df = crime_df.withColumn("date", to_date(col("occurred_on_date")))

print("Crime data cleaned & normalized to lowercase.")

# ---------------------------------------------------------
# 5. Load Neighborhood GeoJSON
# ---------------------------------------------------------
neigh_raw = (
    spark.read
    .option("multiline", "true")
    .json(NEIGH_PATH)
)

neigh = neigh_raw.select(explode("features").alias("feature"))

neigh = neigh.select(
    col("feature.properties.Name").alias("neighborhood"),
    col("feature.properties.OBJECTID").alias("area_id")
)

neigh = neigh.select(
    lower(col("neighborhood")).alias("neighborhood"),
    col("area_id").cast("int").alias("area_id")
)

print("Neighborhood geojson loaded.")

# ---------------------------------------------------------
# 6. Join All Data
# ---------------------------------------------------------
df = crime_df.join(weather_df, on="date", how="left")

df = df.join(
    neigh,
    df.reporting_area.cast("int") == neigh.area_id,
    "left"
)

print("Data joined successfully.")
print("Columns after join:", df.columns)

# ---------------------------------------------------------
# 7. Feature Engineering
# ---------------------------------------------------------
df = df.withColumn(
    "temp_bucket",
    when(col("high_f") < 32, "freezing")
    .when(col("high_f") < 50, "cold")
    .when(col("high_f") < 70, "cool")
    .when(col("high_f") < 85, "warm")
    .otherwise("hot")
)

df = df.withColumn("has_precip", when(col("precip_inch") > 0, 1).otherwise(0))
df = df.withColumn("has_snow", when(col("snow_inch") > 0, 1).otherwise(0))

print("Feature engineering completed.")

# ---------------------------------------------------------
# 8. Prepare ML Dataset (Simple, EXISTING columns only)
# ---------------------------------------------------------
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Label indexers
offense_indexer = StringIndexer(
    inputCol="offense_code_group",
    outputCol="label_offense",
    handleInvalid="skip"
)

district_indexer = StringIndexer(
    inputCol="district",
    outputCol="label_district",
    handleInvalid="skip"
)

df_ml = offense_indexer.fit(df).transform(df)
df_ml = district_indexer.fit(df_ml).transform(df_ml)

# Only use columns that REALLY exist
FEATURE_COLS = [
    "hour",
    "high_f",
    "low_f",
    "precip_inch",
    "snow_inch",
    "has_precip",
    "has_snow",
    "lat",
    "long"
]

assembler = VectorAssembler(
    inputCols=FEATURE_COLS,
    outputCol="features"
)

for c in FEATURE_COLS:
    df_ml = df_ml.withColumn(c, col(c).cast("double"))

df_ml = df_ml.fillna(0, FEATURE_COLS)

df_ml = assembler.transform(df_ml).select(
    "features",
    "label_offense",
    "label_district",
    "hour",
    "neighborhood",
    "temp_bucket",
    "district"
)

print("ML dataset assembled successfully.")

# ---------------------------------------------------------
# 9. Train/Test Split
# ---------------------------------------------------------
train, test = df_ml.randomSplit([0.7, 0.3], seed=42)

# ---------------------------------------------------------
# 10. Model 1: RandomForest → offense_code_group
# ---------------------------------------------------------
rf_model = RandomForestClassifier(
    featuresCol="features",
    labelCol="label_offense",
    numTrees=80,
    maxDepth=12
).fit(train)

rf_preds = rf_model.transform(test)

# ---------------------------------------------------------
# 11. Model 2: Logistic Regression → district
# ---------------------------------------------------------
lr_model = LogisticRegression(
    featuresCol="features",
    labelCol="label_district",
    maxIter=60
).fit(train)

lr_preds = lr_model.transform(test)

# ---------------------------------------------------------
# 12. Evaluation
# ---------------------------------------------------------
eval_acc = MulticlassClassificationEvaluator(metricName="accuracy")
eval_f1 = MulticlassClassificationEvaluator(metricName="f1")

# Evaluation requires columns named "label" and "prediction"
rf_eval_df = rf_preds.select(col("label_offense").alias("label"), "prediction")
lr_eval_df = lr_preds.select(col("label_district").alias("label"), "prediction")

rf_acc = eval_acc.evaluate(rf_eval_df)
rf_f1 = eval_f1.evaluate(rf_eval_df)

lr_acc = eval_acc.evaluate(lr_eval_df)
lr_f1 = eval_f1.evaluate(lr_eval_df)


print("ML evaluation completed.")

# ---------------------------------------------------------
# 13. Offense Group Distribution
# ---------------------------------------------------------
offense_dist = (
    df.groupBy("offense_code_group")
    .agg(count("*").alias("count"))
    .orderBy(col("count").desc())
)

# ---------------------------------------------------------
# 14. Forecasting — Predefined Temperatures
# ---------------------------------------------------------
PREDEFINED_TEMPS = [20, 32, 45, 60, 75, 90]

forecast_rows = []
for t in PREDEFINED_TEMPS:
    # Include ALL feature columns expected by assembler
    synthetic = spark.createDataFrame(
        [(t, t - 5, 0.0, 0.0, 0, 0, 12, 0.0, 0.0)],
        ["high_f", "low_f", "precip_inch", "snow_inch",
         "has_precip", "has_snow", "hour", "lat", "long"]
    )

    synthetic = assembler.transform(synthetic)

    pred_off = rf_model.transform(synthetic).select("prediction").first()[0]
    pred_dist = lr_model.transform(synthetic).select("prediction").first()[0]

    forecast_rows.append((t, float(pred_off), float(pred_dist)))

forecast_df = spark.createDataFrame(
    forecast_rows,
    ["temperature", "predicted_offense_label", "predicted_district_label"]
)

# ---------------------------------------------------------
# 15. Historical Hotspots (Top 15)
# ---------------------------------------------------------
historical = (
    df.groupBy("district", "neighborhood", "hour", "temp_bucket")
    .agg(count("*").alias("crime_count"))
    .orderBy(col("crime_count").desc())
    .limit(15)
)

# ---------------------------------------------------------
# 16. Sample Rows
# ---------------------------------------------------------
sample_rows = df.limit(10).collect()

# ---------------------------------------------------------
# 17. Save Results to Local Text File
# ---------------------------------------------------------
lines = []
lines.append("=== CRIME + WEATHER ML FORECASTING RESULTS ===\n")

lines.append("\n--- RandomForest (Predict offense_code_group) ---")
lines.append(f"Accuracy: {rf_acc}")
lines.append(f"Macro F1: {rf_f1}")

lines.append("\n--- LogisticRegression (Predict district) ---")
lines.append(f"Accuracy: {lr_acc}")
lines.append(f"Macro F1: {lr_f1}")

lines.append("\n--- OFFENSE_CODE_GROUP DISTRIBUTION ---")
for row in offense_dist.collect():
    lines.append(f"{row['offense_code_group']} | {row['count']}")

lines.append("\n--- FORECASTS FOR PREDEFINED TEMPERATURES ---")
for r in forecast_rows:
    lines.append(f"Temp {r[0]}°F → Offense Label: {r[1]} | District Label: {r[2]}")

lines.append("\n--- TOP 15 HISTORICAL HOTSPOTS ---")
for row in historical.collect():
    lines.append(
        f"DISTRICT={row['district']} | Neighborhood={row['neighborhood']} "
        f"| Hour={row['hour']} | Temp={row['temp_bucket']} | Count={row['crime_count']}"
    )

lines.append("\n--- SAMPLE ROWS (FIRST 10) ---")
for r in sample_rows:
    lines.append(str(r.asDict()))

lines.append("\n--- FULL DATA SCHEMA (TEXT) ---")
schema_str = df.schema.simpleString()
lines.append(schema_str)

with open(LOCAL_OUTPUT, "w") as f:
    for l in lines:
        f.write(str(l) + "\n")

print("\nLocal output written to", LOCAL_OUTPUT)

# ---------------------------------------------------------
# 18. Upload to GCS
# ---------------------------------------------------------
client = storage.Client()
bucket = client.bucket(GCS_BUCKET)
blob = bucket.blob(GCS_OUTPUT_PATH)
blob.upload_from_filename(LOCAL_OUTPUT)

print(f"\nUploaded to: gs://{GCS_BUCKET}/{GCS_OUTPUT_PATH}")

spark.stop()
print("\n=== JOB COMPLETED SUCCESSFULLY ===")
