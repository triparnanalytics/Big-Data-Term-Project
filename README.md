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
---
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
---
## Installation & Environment Setup
1. Clone the repository
---
git clone https://github.com/<your-username>/boston-crime-weather.git
cd boston-crime-weather
---
3. Create a virtual environment
---
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
---
4. Install Python dependencies
---
pip install -r requirements.txt
---
5. Install & Configure Apache Spark
---
Download Spark 3.4+

Install Hadoop binaries if on Windows
---
Ensure SPARK_HOME and PATH are set correctly:
---
export SPARK_HOME="/path/to/spark"
export PATH="$SPARK_HOME/bin:$PATH"
---
## Quick Start
Run the full notebook
---
jupyter notebook notebooks/Crime_and_Weather_Analysis.ipynb
---
Or run individual preprocessing steps

Crime preprocessing:
---
spark-submit src/preprocess_crime.py
---

Weather preprocessing:
---
spark-submit src/preprocess_weather.py
---

Join + aggregation:
'''
spark-submit src/join_and_aggregate.py
'''

Train Random Forest model:
---
spark-submit src/model_random_forest.py
---
