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
