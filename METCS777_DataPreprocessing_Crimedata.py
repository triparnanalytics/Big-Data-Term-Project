"""
PySpark Crime Data Preprocessing
Clean, efficient preprocessing with all key features included
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

# CONFIGURATION

INPUT_PATH = "/Users/keerthiuppalapati/Desktop/crime_sample_1000.csv"
OUTPUT_PATH = "/Users/keerthiuppalapati/Desktop/preprocessed_crime_data"

# PREPROCESSING PIPELINE

print("PySpark Crime Data Preprocessing Pipeline")

# Create Spark Session
spark = SparkSession.builder \
    .appName("CrimePreprocessing") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Load Data
print("\nLoading data:")
df = spark.read.csv(INPUT_PATH, header=True, inferSchema=True, nullValue="", nanValue="NaN")
print(f"Loaded {df.count()} records with {len(df.columns)} columns")

# Remove Duplicates
print("\nRemoving duplicates")
initial_count = df.count()
df = df.dropDuplicates()
print(f"Removed {initial_count - df.count()} duplicate records")

# Clean String Columns
print("\nCleaning string columns")
string_cols = ['INCIDENT_NUMBER', 'OFFENSE_CODE_GROUP', 'OFFENSE_DESCRIPTION', 
               'DISTRICT', 'UCR_PART', 'STREET', 'DAY_OF_WEEK']
for col_name in string_cols:
    if col_name in df.columns:
        df = df.withColumn(col_name, trim(col(col_name)))
print(f"Cleaned {len(string_cols)} string columns")

# Convert Data Types & Handle Missing Values
print("\nConverting data types and handling missing values")

# Convert timestamp
df = df.withColumn("OCCURRED_ON_DATE", to_timestamp(col("OCCURRED_ON_DATE"), "yyyy-MM-dd HH:mm:ss"))

# Create SHOOTING binary flag
df = df.withColumn("SHOOTING_FLAG", when(upper(col("SHOOTING")) == "Y", 1).otherwise(0))

# Handle coordinates
df = df.withColumn("Lat", 
                   when((col("Lat") == 0) | (col("Lat").isNull()), None)
                   .otherwise(col("Lat")))
df = df.withColumn("Long", 
                   when((col("Long") == 0) | (col("Long").isNull()), None)
                   .otherwise(col("Long")))

# Impute coordinates
lat_mean = df.filter(col("Lat").isNotNull()).agg(mean("Lat")).first()[0]
long_mean = df.filter(col("Long").isNotNull()).agg(mean("Long")).first()[0]

df = df.withColumn("Lat_Clean", coalesce(col("Lat"), lit(lat_mean)))
df = df.withColumn("Long_Clean", coalesce(col("Long"), lit(long_mean)))
df = df.withColumn("Lat_Missing_Flag", when(col("Lat").isNull(), 1).otherwise(0))
df = df.withColumn("Long_Missing_Flag", when(col("Long").isNull(), 1).otherwise(0))

# Fill missing categorical values
df = df.fillna({
    'DISTRICT': 'UNKNOWN',
    'OFFENSE_CODE_GROUP': 'Other',
    'UCR_PART': 'Unknown',
    'STREET': 'UNKNOWN',
    'REPORTING_AREA': 'UNKNOWN'
})

print(f"Converted types, imputed coordinates (Lat: {lat_mean:.6f}, Long: {long_mean:.6f})")

# 6. Feature Engineering
print("\nEngineering features")

# Time of day
df = df.withColumn("TIME_OF_DAY",
    when((col("HOUR") >= 6) & (col("HOUR") < 12), "Morning")
    .when((col("HOUR") >= 12) & (col("HOUR") < 17), "Afternoon")
    .when((col("HOUR") >= 17) & (col("HOUR") < 21), "Evening")
    .otherwise("Night"))

# Weekend indicator
df = df.withColumn("IS_WEEKEND",
    when(col("DAY_OF_WEEK").isin(["Saturday", "Sunday"]), 1).otherwise(0))

# Crime severity
df = df.withColumn("CRIME_SEVERITY",
    when(col("UCR_PART") == "Part One", 3)
    .when(col("UCR_PART") == "Part Two", 2)
    .when(col("UCR_PART") == "Part Three", 1)
    .otherwise(0))

# Season
df = df.withColumn("SEASON",
    when(col("MONTH").isin([12, 1, 2]), "Winter")
    .when(col("MONTH").isin([3, 4, 5]), "Spring")
    .when(col("MONTH").isin([6, 7, 8]), "Summer")
    .otherwise("Fall"))

# Quarter
df = df.withColumn("QUARTER",
    when(col("MONTH").isin([1, 2, 3]), 1)
    .when(col("MONTH").isin([4, 5, 6]), 2)
    .when(col("MONTH").isin([7, 8, 9]), 3)
    .otherwise(4))

# Day number
df = df.withColumn("DAY_NUMBER",
    when(col("DAY_OF_WEEK") == "Monday", 1)
    .when(col("DAY_OF_WEEK") == "Tuesday", 2)
    .when(col("DAY_OF_WEEK") == "Wednesday", 3)
    .when(col("DAY_OF_WEEK") == "Thursday", 4)
    .when(col("DAY_OF_WEEK") == "Friday", 5)
    .when(col("DAY_OF_WEEK") == "Saturday", 6)
    .when(col("DAY_OF_WEEK") == "Sunday", 7)
    .otherwise(0))

# Offense category
df = df.withColumn("OFFENSE_CATEGORY",
    when(col("OFFENSE_CODE_GROUP").contains("Assault"), "Violent")
    .when(col("OFFENSE_CODE_GROUP").contains("Robbery"), "Violent")
    .when(col("OFFENSE_CODE_GROUP").contains("Homicide"), "Violent")
    .when(col("OFFENSE_CODE_GROUP").contains("Larceny"), "Property")
    .when(col("OFFENSE_CODE_GROUP").contains("Burglary"), "Property")
    .when(col("OFFENSE_CODE_GROUP").contains("Vandalism"), "Property")
    .when(col("OFFENSE_CODE_GROUP").contains("Auto Theft"), "Property")
    .when(col("OFFENSE_CODE_GROUP").contains("Drug"), "Drug")
    .when(col("OFFENSE_CODE_GROUP").contains("Motor Vehicle"), "Traffic")
    .otherwise("Other"))

# High risk hour
df = df.withColumn("HIGH_RISK_HOUR",
    when(col("HOUR").isin([0, 1, 2, 3, 22, 23]), 1).otherwise(0))

print("Created 9 new features")

# 7. Encode Categorical Variables
print("\nEncoding categorical variables")
categorical_cols = ['DISTRICT', 'TIME_OF_DAY', 'SEASON', 'OFFENSE_CATEGORY', 'UCR_PART']

indexers = []
encoders = []

for col_name in categorical_cols:
    indexer = StringIndexer(inputCol=col_name, 
                            outputCol=f"{col_name}_Index",
                            handleInvalid="keep")
    encoder = OneHotEncoder(inputCol=f"{col_name}_Index",
                           outputCol=f"{col_name}_Encoded",
                           dropLast=True)
    indexers.append(indexer)
    encoders.append(encoder)

pipeline = Pipeline(stages=indexers + encoders)
df = pipeline.fit(df).transform(df)
print(f"Encoded {len(categorical_cols)} categorical columns")

# 8. Scale Numeric Features
print("\nScaling numeric features")
numeric_cols = ['OFFENSE_CODE', 'YEAR', 'MONTH', 'HOUR', 'QUARTER',
                'Lat_Clean', 'Long_Clean', 'DAY_NUMBER', 'CRIME_SEVERITY']

available_cols = [c for c in numeric_cols if c in df.columns]

assembler = VectorAssembler(inputCols=available_cols,
                            outputCol="numeric_features",
                            handleInvalid="skip")

scaler = StandardScaler(inputCol="numeric_features",
                       outputCol="scaled_features",
                       withMean=True,
                       withStd=True)

scale_pipeline = Pipeline(stages=[assembler, scaler])
df = scale_pipeline.fit(df).transform(df)
print(f"Scaled {len(available_cols)} numeric features")

# 9. Display Results
print("\nPreprocessing complete!")
print("PREPROCESSING SUMMARY")
print(f"Final record count: {df.count()}")
print(f"Final column count: {len(df.columns)}")

print("\nAll Features in Dataset")
print("Original Columns:", df.columns[:17])
print("\nNew Features Created:")
new_features = [
    'SHOOTING_FLAG', 'Lat_Clean', 'Long_Clean', 'Lat_Missing_Flag', 
    'Long_Missing_Flag', 'TIME_OF_DAY', 'IS_WEEKEND', 'CRIME_SEVERITY',
    'SEASON', 'QUARTER', 'DAY_NUMBER', 'OFFENSE_CATEGORY', 'HIGH_RISK_HOUR'
]
for feat in new_features:
    print(f"   {feat}")

print("\nEncoded Columns:")
for col_name in categorical_cols:
    print(f"   {col_name}_Index")
    print(f"   {col_name}_Encoded")

print("\nScaled Features:")
print(f"   numeric_features (vector of {len(available_cols)} features)")
print(f"   scaled_features (standardized)")

print("\nSample of Preprocessed Data (20 rows)")
df.select('INCIDENT_NUMBER', 'OFFENSE_CATEGORY', 'TIME_OF_DAY', 
          'CRIME_SEVERITY', 'IS_WEEKEND', 'SHOOTING_FLAG', 'SEASON', 'QUARTER').show(20, truncate=False)

print("\nDistribution of Offense Categories")
df.groupBy('OFFENSE_CATEGORY').count().orderBy(col('count').desc()).show()

print("\nDistribution of Time of Day")
df.groupBy('TIME_OF_DAY').count().orderBy(col('count').desc()).show()

# 10. Save Data
print("\nSaving preprocessed data")

# Save complete data as Parquet (includes ALL features and vectors)
print(f"\nSaving complete dataset (Parquet)")
df.write.mode("overwrite").parquet(OUTPUT_PATH + "_COMPLETE.parquet")

print(f"{OUTPUT_PATH}_COMPLETE.parquet/")
print(f"    All {len(df.columns)} columns (includes encoded vectors for ML)")

# Save human-readable data as CSV (excludes complex vector types)
print(f"\nSaving human-readable dataset (CSV)")

# Select all columns except vector types (they can't be saved to CSV)
csv_compatible_cols = [c for c in df.columns 
                       if not c.endswith('_Encoded') 
                       and 'features' not in c.lower()
                       and c != 'Location']

csv_df = df.select(csv_compatible_cols)

# Save as multiple CSV files
csv_df.write.mode("overwrite").option("header", "true").csv(OUTPUT_PATH + "_CSV")
print(f" {OUTPUT_PATH}_CSV/")

# Save as single CSV file
csv_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(OUTPUT_PATH + "_SINGLE_CSV")
print(f" {OUTPUT_PATH}_SINGLE_CSV/")
print(f"    {len(csv_compatible_cols)} columns")

print("PREPROCESSING COMPLETE")

print("\n Summary:")
print(f"   • Original features: 17")
print(f"   • New features created: {len(new_features)}")
print(f"   • Encoded features: {len(categorical_cols) * 2}")
print(f"   • Total columns: {len(df.columns)}")
print(f"   • Columns in CSV: {len(csv_compatible_cols)} (vectors excluded)")

print("\n Key Features Included:")
print("    All original columns")
print("    SHOOTING_FLAG (binary)")
print("    Cleaned coordinates (Lat_Clean, Long_Clean)")
print("    Missing indicators (Lat_Missing_Flag, Long_Missing_Flag)")
print("    TIME_OF_DAY, SEASON, QUARTER")
print("    OFFENSE_CATEGORY, CRIME_SEVERITY")
print("    IS_WEEKEND, HIGH_RISK_HOUR, DAY_NUMBER")
print("    Categorical encodings (Index + OneHot)")
print("    Scaled numeric features")

print("\n Files Created:")
print(f"   1. {OUTPUT_PATH}_COMPLETE.parquet/ - Full dataset with vectors")
print(f"   2. {OUTPUT_PATH}_CSV/ - CSV format (multiple files)")
print(f"   3. {OUTPUT_PATH}_SINGLE_CSV/ - Single CSV file")

print("\n All preprocessing complete")
spark.stop()