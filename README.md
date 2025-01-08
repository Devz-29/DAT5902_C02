# DAT5902_C02
# Author: [Dev Arjun Prabhakar]
# Date: 2025-01-08

## Description

This repository contains the code and analysis for the **DAT5902 Final Project**, which explores global CO₂ emissions trends and their relationship with GDP per capita. The analysis includes:

- Data cleaning  of CO₂ emissions and GDP datasets.
- Exploratory Data Analysis (EDA) with visualizations, including trends, histograms, and heatmaps.
- Linear regression to assess the relationship between GDP and CO₂ emissions.
- Geospatial analysis using GeoPandas to create heatmaps of CO₂ emissions by country.

The findings highlight key patterns in CO₂ emissions, the impact of economic growth, and regional disparities over time.

## Project Structure

The repository is organized as follows:

- **`data/`**: Contains the datasets used for the analysis.
  - `co2_emission.csv`: Annual CO₂ emissions data by country.
  - `gdp-per-capita-worldbank.csv`: GDP per capita data from the World Bank.
  - `countries.geo.json`: Geographic boundary data for mapping.

- **`scripts/`**: Python scripts for analysis.
  - `main.py`: Main script for running the full analysis.
  - `eda.py`: Exploratory Data Analysis and visualization code.
  - `linear_regression.py`: Linear regression model for GDP vs. CO₂ emissions.
  - `heatmap.py`: Code to generate geospatial heatmaps.

- **`tests/`**: Contains unit tests for validating the analysis.
  - `Unit_Test.py`: Unit tests for data validation and model performance.

- **Figures**: Output visualizations and saved plots, such as:
  - `CO2_emissions.png`
  - `Linear Regression Model GDP vs. C02.png`
  - `co2_heatmap_2016.png`
 - `co2_heatmap_1975.png`
 - `Average_CO2.png`
 - `time_series_CO2_emissions.png`

- **`README.md`**: Repository documentation (this file).
