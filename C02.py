import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as scp

df = pd.read_csv('co2_emission.csv') # load the data

print(df.head())       # Preview the first few rows
print(df.info())       # Get an overview of columns, data types, and missing values
print(df.describe())   # Summary statistics for numerical columns

print(df.isnull().sum()) # Check for missing values

print(df.dtypes) # Check types of columns
print(df.columns)

df = df.rename(columns={'Annual CO₂ emissions (tonnes )': 'CO2_emission'}) # rename columns for better readibility

print(df.info())

import seaborn as sns


# -------------------------
# 1. Distribution of CO2 emissions
sns.histplot(df['CO2_emission'], kde=True, bins=30)  # Add `kde` and `bins` for better visualization
plt.title('Histogram of CO2 Emissions')
plt.xlabel('CO2 Emissions')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('CO2_emissions.png')
plt.show()

# -------------------------
# 2. Correlation Matrix
# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Compute the correlation matrix
correlation_matrix = numeric_df.corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig('Heatmap.png')
plt.show()

# -------------------------
# 3. Average CO2 Emissions Over Time
# Group by 'Year' and calculate the mean CO2 emissions
df_grouped = df.groupby('Year')['CO2_emission'].mean()

# Plot the results
plt.figure(figsize=(10, 5))
df_grouped.plot(kind='line', color='blue', marker='o')
plt.title('Average CO₂ Emissions Over Time')
plt.xlabel('Year')
plt.ylabel('CO₂ Emissions')
plt.grid(True)
plt.savefig('Average_CO2.png')
plt.show()

