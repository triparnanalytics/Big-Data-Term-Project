# Boston Crime & Weather Analysis (PySpark + ML)
Exploring how temperature, precipitation, and seasonal patterns influence crime in Boston (2015–2018).

## Abstract
This project analyzes how Boston crime patterns relate to weather conditions using PySpark-based preprocessing, correlation analysis, and machine learning. Crime incidents are linked with daily weather data, enriched with temporal and temperature features, and used to study the effects of heat, cold, precipitation, and snow. A Random Forest classifier predicts offense category using time, location, and weather inputs, while risk summaries identify high-risk neighborhood–hour–temperature combinations.

## Project Overview

The goal of this project is to quantify how temperature, precipitation, and snow are associated with crime levels in Boston and to build an operational tool that highlights high-risk combinations of area, time, and temperature. Using Spark, the workflow:
- Cleans and enriches raw crime data with temporal, severity, and behavioral features.
- Cleans daily Boston weather, handles trace/missing values, and engineers seasonal features.
- Joins daily crime with weather, computes rolling correlations, and runs linear regression tests.
- Trains a Random Forest classifier to predict offense category.
- Produces interpretable risk tables identifying district-hour-temperature hotspots.
## Repository Structure
```bash
boston-crime-weather/
│
├── data/
│   ├── crime.csv
│   ├── boston-weather.csv
│   └── processed/
│
├── notebooks/
│   └── Crime_and_Weather_Analysis.ipynb
│
├── src/
│   ├── preprocess_crime.py
│   ├── preprocess_weather.py
│   ├── join_and_aggregate.py
│   └── model_random_forest.py
│
├── results/
│   ├── rolling_correlations.png
│   ├── seasonal_patterns.png
│   ├── regression_summary.txt
│   ├── rf_confusion_matrix.png
│   └── risk_tables/
│
├── README.md
└── requirements.txt
```
## Installation & Environment Setup
1. Clone the repository
```bash
git clone https://github.com/<your-username>/boston-crime-weather.git
cd boston-crime-weather
```
3. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```
4. Install Python dependencies
```bash
pip install -r requirements.txt
```
5. Install & Configure Apache Spark

Download Spark 3.4+

Install Hadoop binaries if on Windows

Ensure SPARK_HOME and PATH are set correctly:
```bash
export SPARK_HOME="/path/to/spark"
export PATH="$SPARK_HOME/bin:$PATH"
```
## Quick Start
Run the full notebook
```bash
jupyter notebook notebooks/Crime_and_Weather_Analysis.ipynb
```
Or run individual preprocessing steps

Crime preprocessing:
```bash
spark-submit src/preprocess_crime.py
```
Weather preprocessing:
```bash
spark-submit src/preprocess_weather.py
```
Join + aggregation:
```bash
spark-submit src/join_and_aggregate.py
```
Train Random Forest model:
```bash
spark-submit src/model_random_forest.py
```
## Dataset Information
## Crime Dataset (Boston Police Incidents)
- Source: Boston Police Department 2015–2018
- Incident records: ~319k
- Key variables: offense codes, district, hour, UCR part, coordinates

## Engineered features:
- OFFENSECATEGORY (Violent, Property, Drug, Traffic, Other)
- CRIMESEVERITY (UCR part–based score)
- Temporal features: season, weekend, quarter, time-of-day
- Behavior indicators: HIGHRISKHOUR, SHOOTINGFLAG

## Weather Dataset (Boston Daily Weather)
- Daily high/low temperature
- Precipitation (with trace values handled)
- Snowfall + snow depth
- Derived: temprange, season, hasprecipitation, hassnow
## Neighborhood / District Mapping:
District codes (A1, B2, C11, etc.) mapped to readable names (Downtown, Roxbury, Dorchester, etc.) used for area-based analysis.

## File Description
## Crime_and_Weather_Analysis.ipynb

End-to-end workflow containing:
- Crime preprocessing + feature engineering
- Weather preprocessing
- Daily join + rolling averages
- Correlation analysis
- Regression modeling
- Random Forest classification
- Risk summarization by neighborhood, hour, and temperature bucket

## Dataset Explanation
## Crime Features
- Event-level: offense group, district, hour, day-of-week, UCR part
-Enriched:
- Offense category grouping
- Severity scoring
- Time-of-day buckets (Morning/Afternoon/Evening/Night)
- Weekend, season, and quarter
- Late-night high-risk hours (0–3, 22–23)

## Weather Features
- Clean high/low temp, precipitation, snow
- Derived temperature range and seasonal labels
- Precipitation/snow binary indicators

## Joined Daily Dataset
Includes:
- totalcrimes, violentcrimes, propertycrimes, shootings
- highf, lowf, precipinch, snowinch
- 30-day rolling averages for correlation analysis

## Preprocessing Stages
## Crime Preprocessing
- Drop duplicates
- Clean key string columns
- Parse timestamps → crimedate, year, month, day, hour
- Impute missing lat/long
- Fill missing categorical labels
- Engineer all temporal + severity + category features
- Encode categorical fields for ML
- Produce reusable clean CSV

## Weather Preprocessing
- Convert “T”→0.001 and missing markers
- Normalize column names
- Parse date field
- Impute missing numeric values
- Engineer temperature range + seasonal flags

## Crime–Weather Join

- Aggregate crimes daily
- Join on date
- Reindex in pandas to fill missing days
- Compute rolling windows for correlation

## Results & Observations
## Correlation Analysis
- Crime rises with temperature:
-- High temp correlation ≈ 0.71
-- Low temp correlation ≈ 0.70
- Precipitation reduces crime (≈ −0.29 to −0.36)
- Violent and property crimes show similar temperature sensitivity

## Seasonal & Temperature Patterns
- Freezing days → lowest crime (~233/day)
- Crime increases through Cool → Warm → Hot
- Summer > Fall > Spring > Winter

## High-Risk Area/Time/Temperature Hotspots
- Late-night Downtown & South End in Cool/Warm conditions
- Elevated violent-crime fractions in East Boston, Mattapan during evening/night
- Hot days amplify late-night crime risk across multiple districts

## Model Evaluation
## Linear Regression
- Interpretable
- Statistically meaningful coefficients
- Limited variance explained (weather alone insufficient)

## Random Forest
- Handles mixed data types & nonlinearities
- Scales well to 300K+ records
- Accuracy modest due to overlapping crime categories
- Captures nonlinear interactions between time, district, and weather
