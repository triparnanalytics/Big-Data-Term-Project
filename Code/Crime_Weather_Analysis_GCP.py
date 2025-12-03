############################################################
# Crime + Weather ML Forecasting Pipeline (GCP Dataproc)
# Target = offense_code_group (classification)
# + Daily Linear Regression (crime ~ weather)
# + Rolling 30-day correlation analysis
############################################################

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from google.cloud import storage

import pandas as pd
import numpy as np

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator

# ---------------------------------------------------------
# 1. Spark Session
# ---------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("CrimeWeatherForecasting")
    .config("spark.sql.session.timeZone", "UTC")
    # reduce risk of huge broadcast joins
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.broadcastJoinThreshold", "-1")
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
GCS_OUTPUT_PATH = "output/results_gcp.txt"

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
# 7. Feature Engineering (incident-level)
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
df = df.withColumn("temp_range", col("high_f") - col("low_f"))

# Crime type classification for violent vs property
df = df.withColumn(
    "crime_type",
    when(col("offense_code_group").isNull(), "other")
    .when(col("offense_code_group").rlike("(?i)assault|robbery|weapon|homicide|violence"), "violent")
    .when(col("offense_code_group").rlike("(?i)larceny|theft|burglary|property|vandalism|breaking"), "property")
    .otherwise("other")
)

print("Feature engineering completed.")

# ---------------------------------------------------------
# 7A. DAILY AGGREGATES FOR CORRELATION & LINEAR REGRESSION
# ---------------------------------------------------------

# Daily crime aggregates
crime_daily = (
    df.groupBy("date")
      .agg(
          count("*").alias("total_crimes"),
          sum(when(col("crime_type") == "violent", 1).otherwise(0)).alias("violent_crimes"),
          sum(when(col("crime_type") == "property", 1).otherwise(0)).alias("property_crimes"),
          sum(when(col("shooting") == "Y", 1).otherwise(0)).alias("shootings")
      )
      .withColumnRenamed("date", "crime_date")
)

# Daily weather aggregates
weather_daily = (
    weather_df.select("date", "high_f", "low_f", "precip_inch")
              .withColumn("temp_range", col("high_f") - col("low_f"))
)

# Join daily crime + weather
daily_joined = (
    crime_daily.join(
        weather_daily,
        crime_daily.crime_date == weather_daily.date,
        "inner"
    )
    .select(
        "crime_date",
        "total_crimes",
        "violent_crimes",
        "property_crimes",
        "shootings",
        "high_f",
        "low_f",
        "temp_range",
        "precip_inch"
    )
)

# Simple Pearson correlations (no rolling)
corr_high_simple = daily_joined.stat.corr("high_f", "total_crimes")
corr_low_simple = daily_joined.stat.corr("low_f", "total_crimes")

print(f"Simple correlation (Total Crimes vs High Temp): {corr_high_simple}")
print(f"Simple correlation (Total Crimes vs Low Temp): {corr_low_simple}")

# ---------------------------------------------------------
# 7B. 30-DAY ROLLING WINDOW CORRELATION ANALYSIS
# ---------------------------------------------------------

print("=" * 70)
print("CORRELATION ANALYSIS (30-Day Rolling Windows)")
print("=" * 70)

crime_daily_pd = (
    crime_daily
    .select("crime_date", "total_crimes", "violent_crimes", "property_crimes", "shootings")
    .orderBy("crime_date")
    .toPandas()
)

if not crime_daily_pd.empty:
    crime_daily_pd["crime_date"] = pd.to_datetime(crime_daily_pd["crime_date"])
    crime_daily_pd = crime_daily_pd.set_index("crime_date")

    full_date_range = pd.date_range(
        start=crime_daily_pd.index.min(),
        end=crime_daily_pd.index.max(),
        freq="D"
    )

    crime_daily_extended = crime_daily_pd.reindex(full_date_range)

    window = 30

    crimes_total_rolling = crime_daily_extended["total_crimes"].rolling(window=window, min_periods=1).mean().dropna()
    crimes_violent_rolling = crime_daily_extended["violent_crimes"].rolling(window=window, min_periods=1).mean().dropna()
    crimes_property_rolling = crime_daily_extended["property_crimes"].rolling(window=window, min_periods=1).mean().dropna()
    crimes_shootings_rolling = crime_daily_extended["shootings"].rolling(window=window, min_periods=1).mean().dropna()

    # Weather extended from daily weather
    weather_pd = (
        weather_daily
        .select("date", "high_f", "low_f", "temp_range", "precip_inch")
        .orderBy("date")
        .toPandas()
    )
    weather_pd["date"] = pd.to_datetime(weather_pd["date"])
    weather_pd = weather_pd.set_index("date")
    weather_extended = weather_pd.reindex(full_date_range)

    for col_name in ["high_f", "low_f", "temp_range", "precip_inch"]:
        weather_extended[col_name] = pd.to_numeric(weather_extended[col_name], errors="coerce").interpolate()

    weather_high_rolling = weather_extended["high_f"].rolling(window=window, min_periods=1).mean()
    weather_low_rolling = weather_extended["low_f"].rolling(window=window, min_periods=1).mean()
    weather_range_rolling = weather_extended["temp_range"].rolling(window=window, min_periods=1).mean()
    weather_precip_rolling = weather_extended["precip_inch"].rolling(window=window, min_periods=1).mean()

    # Align to crimes_total_rolling index
    weather_high_aligned = weather_high_rolling.reindex(crimes_total_rolling.index)
    weather_low_aligned = weather_low_rolling.reindex(crimes_total_rolling.index)
    weather_range_aligned = weather_range_rolling.reindex(crimes_total_rolling.index)
    weather_precip_aligned = weather_precip_rolling.reindex(crimes_total_rolling.index)

    correlations = {}
    if len(crimes_total_rolling) > 1:
        correlations = {
            "Total Crimes vs High Temp": np.corrcoef(crimes_total_rolling, weather_high_aligned)[0, 1],
            "Total Crimes vs Low Temp": np.corrcoef(crimes_total_rolling, weather_low_aligned)[0, 1],
            "Total Crimes vs Temp Range": np.corrcoef(crimes_total_rolling, weather_range_aligned)[0, 1],
            "Violent Crimes vs High Temp": np.corrcoef(crimes_violent_rolling, weather_high_aligned)[0, 1],
            "Property Crimes vs High Temp": np.corrcoef(crimes_property_rolling, weather_high_aligned)[0, 1],
            "Shootings vs High Temp": np.corrcoef(crimes_shootings_rolling, weather_high_aligned)[0, 1],
        }

        print("\nCorrelation Coefficients (30-Day Rolling Windows):")
        for key, value in correlations.items():
            print(f"  {key}: {value * 100:.2f}%")
    else:
        correlations = {}
        print("Not enough data for rolling-window correlation analysis.")
else:
    correlations = {}
    full_date_range = pd.DatetimeIndex([])
    print("No daily crime data available for rolling-window correlation analysis.")

print("=" * 70)

# ---------------------------------------------------------
# 8. Prepare ML Dataset (incident-level RF classification)
# ---------------------------------------------------------

offense_indexer = StringIndexer(
    inputCol="offense_code_group",
    outputCol="label_offense",
    handleInvalid="skip"
)

df_ml = offense_indexer.fit(df).transform(df)

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

# Cast features to double + fill nulls
for c in FEATURE_COLS:
    df_ml = df_ml.withColumn(c, col(c).cast("double"))
df_ml = df_ml.fillna(0, FEATURE_COLS)

df_ml = df_ml.select(
    *FEATURE_COLS,
    "label_offense",
    "hour",
    "neighborhood",
    "temp_bucket",
    "district"
)

df_ml = assembler.transform(df_ml).select(
    "features",
    "label_offense",
    "hour",
    "neighborhood",
    "temp_bucket",
    "district"
)

print("ML dataset assembled successfully.")

# ---------------------------------------------------------
# 9. Train/Test Split for RF
# ---------------------------------------------------------
train, test = df_ml.randomSplit([0.7, 0.3], seed=42)

# ---------------------------------------------------------
# 10. Model 1: RandomForest → offense_code_group
# ---------------------------------------------------------
rf_model = RandomForestClassifier(
    featuresCol="features",
    labelCol="label_offense",
    numTrees=25,   # reduced for stability
    maxDepth=8     # reduced for stability
).fit(train)

rf_preds = rf_model.transform(test)

# ---------------------------------------------------------
# 11. Evaluation for RF
# ---------------------------------------------------------
eval_acc = MulticlassClassificationEvaluator(
    metricName="accuracy", labelCol="label", predictionCol="prediction"
)
eval_f1 = MulticlassClassificationEvaluator(
    metricName="f1", labelCol="label", predictionCol="prediction"
)

rf_eval_df = rf_preds.select(col("label_offense").alias("label"), "prediction")

rf_acc = eval_acc.evaluate(rf_eval_df)
rf_f1 = eval_f1.evaluate(rf_eval_df)

print("RF classification evaluation completed.")

# ---------------------------------------------------------
# 12. LINEAR REGRESSION: Daily Total Crimes ~ Weather
# ---------------------------------------------------------

daily_reg = (
    daily_joined
    .select(
        "crime_date",
        "total_crimes",
        "high_f",
        "low_f",
        "temp_range",
        "precip_inch"
    )
    .na.drop()
)

assembler_reg = VectorAssembler(
    inputCols=["high_f", "low_f", "temp_range", "precip_inch"],
    outputCol="features"
)

daily_reg = assembler_reg.transform(daily_reg)

train_reg, test_reg = daily_reg.randomSplit([0.7, 0.3], seed=42)

linreg = LinearRegression(
    featuresCol="features",
    labelCol="total_crimes",
    maxIter=100
).fit(train_reg)

reg_preds = linreg.transform(test_reg)

reg_eval_rmse = RegressionEvaluator(
    metricName="rmse",
    labelCol="total_crimes",
    predictionCol="prediction"
).evaluate(reg_preds)

reg_eval_mae = RegressionEvaluator(
    metricName="mae",
    labelCol="total_crimes",
    predictionCol="prediction"
).evaluate(reg_preds)

reg_eval_r2 = RegressionEvaluator(
    metricName="r2",
    labelCol="total_crimes",
    predictionCol="prediction"
).evaluate(reg_preds)

print("Linear Regression (daily crime vs weather) evaluation completed.")

# ---------------------------------------------------------
# 13. Offense Group Distribution (limit for safety)
# ---------------------------------------------------------
offense_dist = (
    df.groupBy("offense_code_group")
    .agg(count("*").alias("count"))
    .orderBy(col("count").desc())
    .limit(50)
)

# ---------------------------------------------------------
# 14. Forecasting — Predefined Temperatures (offense labels only)
# ---------------------------------------------------------
PREDEFINED_TEMPS = [20, 32, 45, 60, 75, 90]

forecast_rows = []
for t in PREDEFINED_TEMPS:
    synthetic = spark.createDataFrame(
        [(12.0, t, t - 5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
        ["hour", "high_f", "low_f", "precip_inch", "snow_inch",
         "has_precip", "has_snow", "lat", "long"]
    )
    synthetic = assembler.transform(synthetic)
    pred_off = rf_model.transform(synthetic).select("prediction").first()[0]
    forecast_rows.append((t, float(pred_off)))

forecast_df = spark.createDataFrame(
    forecast_rows,
    ["temperature", "predicted_offense_label"]
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

lines.append("\n--- Linear Regression (Daily Total Crimes ~ Weather) ---")
lines.append(f"RMSE: {reg_eval_rmse}")
lines.append(f"MAE: {reg_eval_mae}")
lines.append(f"R^2: {reg_eval_r2}")

lines.append("\n--- SIMPLE CORRELATIONS (Daily Totals) ---")
lines.append(f"Total Crimes vs High Temp: {corr_high_simple}")
lines.append(f"Total Crimes vs Low Temp: {corr_low_simple}")

lines.append("\n--- ROLLING 30-DAY CORRELATIONS (if available) ---")
if correlations:
    for key, value in correlations.items():
        lines.append(f"{key}: {value * 100:.2f}%")
else:
    lines.append("Not enough data for rolling-window correlations.")

lines.append("\n=== OFFENSE_CODE_GROUP DISTRIBUTION (Top 50) ===")
for row in offense_dist.collect():
    lines.append(f"{row['offense_code_group']} | {row['count']}")

lines.append("\n=== FORECASTS FOR PREDEFINED TEMPERATURES (RF, offense labels) ===")
for r in forecast_rows:
    lines.append(f"Temp {r[0]}°F → Offense Label: {r[1]}")

lines.append("\n=== TOP 15 HISTORICAL HOTSPOTS ===")
for row in historical.collect():
    lines.append(
        f"DISTRICT={row['district']} | Neighborhood={row['neighborhood']} "
        f"| Hour={row['hour']} | Temp={row['temp_bucket']} | Count={row['crime_count']}"
    )

lines.append("\n=== SAMPLE ROWS (FIRST 10) ===")
for r in sample_rows:
    lines.append(str(r.asDict()))

lines.append("\n=== FULL DATA SCHEMA (TEXT) ===")
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
