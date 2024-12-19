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

