# Boston Crime & Weather Analysis (PySpark + ML)
Exploring how temperature, precipitation, and seasonal patterns influence crime in Boston (2015–2018).

## Abstract
This project analyzes how Boston crime patterns relate to weather conditions using PySpark-based preprocessing, correlation analysis, and machine learning. Crime incidents are linked with daily weather data, enriched with temporal and temperature features, and used to study the effects of heat, cold, precipitation, and snow. A Random Forest classifier predicts offense category using time, location, and weather inputs, while risk summaries identify high-risk neighborhood–hour–temperature combinations.

## Project Overview
- By: Triparna Kundu, Keerthi Uppalapati & Nidhi Nama
- Project Purpose: Quantify how temperature, precipitation, and snow influence crime levels across Boston neighborhoods, and build a predictive tool to forecast high-risk crime periods and locations.
- Motivation: Understanding the environmental and temporal drivers of crime can help law enforcement and city planners allocate resources efficiently and implement proactive safety measures. Combining weather data with neighborhood-specific crime patterns enables more accurate risk assessment and situational awareness.
- Methods:
  - Clean and enrich raw crime data with temporal (hour, day, season), severity, and behavioral features.
  - Process daily Boston weather data, handle missing or trace values, and engineer seasonal indicators.
  - Integrate crime and weather datasets by neighborhood and date, compute rolling correlations, and run exploratory regression analyses.
  - Train machine learning models, including Random Forest and other classifiers, to forecast offense categories and crime intensity.
  - Generate interpretable risk tables and visualizations highlighting high-risk neighborhood-hour-weather combinations.
- Outcome: A predictive and operational tool that identifies crime hotspots based on weather and temporal patterns, enabling data-driven policing strategies and enhanced public safety.
---
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

## Boston_Neighborhood / District Mapping
- Use the .geoson file to extraxt the district codes (A1, B2, C11, etc.) mapped to readable names (Downtown, Roxbury, Dorchester, etc.) and implement it for area-based analysis.
---
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
  
## Analysis & Modeling
## Correlation Analysis
- Crime rises with temperature:
  - High temp correlation ≈ 0.71
  - Low temp correlation ≈ 0.70
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

---
## Model Evaluation
## Logistic Regression
- Interpretable
- Suitable for binary or categorical outcomes
- Measures the effect of predictors on the probability of an event
- Model fit can be assessed using accuracy.
- Limited predictive power if only weather features are used; additional features improve performance.
  
## Random Forest
- Handles mixed data types & nonlinearities
- Scales well to 300K+ records
- Accuracy modest due to overlapping crime categories
- Captures nonlinear interactions between time, district, and weather
- Define the number of trees and depth to calculate the accuracy of the model.
---

## Repository Structure
```bash
boston-crime-weather/
│
├── data/
│   ├── crime.csv
│   ├── boston-weather.csv
│   └── Boston_Neighbourhoods.geoson
│
├── notebooks/
│   └── Crime_and_Weather_Analysis.ipynb
│
├── src/
│   ├── Crime_and_Weather_Analysis_GCP.py
│ 
├── Output/
│   ├── {output}.png
│   ├── results.txt
├── README.md

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
## How to Run the code
1. Upload Files to Google Cloud Storage
Upload all scripts and datasets to your GCS bucket:
```bash
gs://<bucket_name>/Crime_and_Weather_Analysis_GCP.py
gs://<bucket_name>/boston-weather.csv
gs://<bucket_name>/crime.csv
gs://<bucket_name>/Boston_Neighbours.geojson
```
2. Create a Dataproc Cluster

- Create a Dataproc cluster with PySpark pre-installed.
- Ensure enough nodes for your dataset size.

3. Submit the PySpark Job

Run the analysis on your cluster:
```bash
gcloud dataproc jobs submit pyspark \
    gs://<bucket_name>/Crime_and_Weather_Analysis_GCP.py \
    --cluster <cluster_name> \
    --region <region> \
    -- gs://<bucket_name>/boston-weather.csv gs://<bucket_name>/crime.csv gs://<bucket_name>/output/
```
4. Retrieve Outputs
Results will be stored in your GCS bucket under the output folder:
```bash
gs://<bucket_name>/output/output_results_gcp.txt
```
The results can be downloaded or can be viewed directly from the GCS console.

Or, 

Run the full notebook in your local system
```bash
jupyter notebook notebooks/Crime_and_Weather_Analysis.ipynb
```
---

