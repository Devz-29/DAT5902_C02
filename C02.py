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
sns.histplot(df['CO2_emission'], kde=True, bins=100)  # Add `kde` and `bins` for better visualization
plt.title('Histogram of CO2 Emissions')

# Set the x-axis range and ticks for 1e9 increments
plt.xlim(-3e9, 10e9)  # Set the range of the x-axis
tick_values = np.arange(-3e9, 10.1e9, 1e9)  # Generate tick marks every 1e9
plt.xticks(tick_values, labels=[f'{int(tick/1e9)}B' for tick in tick_values])  # Format ticks as "X B"
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

gdp_df = pd.read_csv('gdp-per-capita-worldbank.csv') #To use as an independant variable to show which countries contribute to CO2 emissions in comparrison to GDP
# -------------------------

gdp_df = gdp_df.rename(columns={'GDP per capita, PPP (constant 2017 international $)': 'GDP'})


merged = gdp_df.merge(df, on='Code', how='inner')  # Merge GDP and CO2 emission data on 'Code'

X = merged['GDP']  # Select GDP as the feature
y = merged['CO2_emission']  # Select CO2 emissions as the target variable

# Verify shapes of the feature and target
print(f"Shape of X: {X.shape}")  # Print shape of X
print(f"Shape of y: {y.shape}")  # Print shape of y

if isinstance(X, pd.Series):  # Check if X is a Series
    X = X.to_frame()  # Convert Series to DataFrame

if len(X.shape) == 1:  # Check if X has only one dimension
    X = X.values.reshape(-1, 1)  # Reshape into (n_samples, 1)

print(f"Shape of X (after conversion): {X.shape}")  # Print shape of X after conversion

from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error  

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=43)

model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))

# Plot the regression line
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title('GDP vs CO2 Emissions')
plt.xlabel('GDP')
plt.ylabel('CO2 Emissions')
plt.show()
plt.savefig('Linear regression Model GDP vs C02')

#displays the geojson map that had to be created in jupyter notebook as ipywidgest is not available otherwise.

import dash
from dash import  dcc
from dash import html 
import pandas as pd
import geopandas as gpd
import folium
from dash.dependencies import Input, Output
import os
import webbrowser
import io
import base64

# Load data
df = pd.read_csv('co2_emission.csv')
df = df.rename(columns={'Annual CO₂ emissions (tonnes )': 'CO2_emission'})
world_geojson = gpd.read_file('countries.geo.json')

# Function to update map based on selected year
def update_map(year):
    year_data = df[df['Year'] == year]
    merged_data = world_geojson.merge(year_data, left_on='name', right_on='Entity', how='left')
    
    # Creates the map
    m = folium.Map(location=[20, 0], zoom_start=2)
    
    folium.Choropleth(
        geo_data=world_geojson,
        name='CO2 Emissions',
        data=merged_data,
        columns=['name', 'CO2_emission'],
        key_on='feature.properties.name',
        fill_color='YlGnBu',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='CO₂ Emissions (tonnes)',
    ).add_to(m)
    
    # Saves to HTML file in a loop
    map_html = f"co2_heatmap_{year}.html"
    m.save(map_html)
    
    return map_html

# Create Dash app
app = dash.Dash(__name__)

# establish Layout of the map
app.layout = html.Div([
    html.H1('CO₂ Emission Map'),
    dcc.Slider(                    # creates the slider to access data from different years
        id='year-slider',
        min=df['Year'].min(),            #minimum year for each country
        max=df['Year'].max(),            # maximum for each country
        value=2015,
        marks={year: str(year) for year in range(df['Year'].min(), df['Year'].max() + 1, 1)},
        step=1,
    ),
    html.Div(id='map-container')
])

# Update the map when slider changes
@app.callback(
    Output('map-container', 'children'),
    [Input('year-slider', 'value')]
)
def update_map_callback(year):
    map_file = update_map(year)  # This will now return the path to the saved HTML file
    return html.Iframe(src=map_file, width='100%', height='600')

# Run the app
if __name__ == '__main__':
    # Run the Dash app in a new process
    import subprocess
    subprocess.Popen(['python3', 'app.py'])

    # Optionally, open the map in the browser
    webbrowser.open('http://127.0.0.1:8050')  # Open the map page in the browser
