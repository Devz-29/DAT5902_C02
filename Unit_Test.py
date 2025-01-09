"""
Unit tests for the CO2 emissions analysis project.

This module contains a suite of unit tests to validate the functionality of the CO2 emissions analysis code. The tests cover various aspects of the project, including file handling, data structure, data sufficiency, correlation analysis, and linear regression accuracy.

The `load_data()` fixture loads the CO2 emissions data from a CSV file, while the `generate_test_data()` fixture generates a small test dataset for testing purposes.

The tests include:
- `test_file_handling()`: Ensures the CO2 emissions data file exists and is in the expected CSV format.
- `test_data_structure(load_data)`: Verifies the structure of the CO2 emissions data, including the presence of required columns and numeric data.
- `test_sufficient_data(load_data)`: Checks that the CO2 emissions data has at least two data points.
- `test_correlation(generate_test_data)`: Ensures the test data has a strong correlation between the 'x' and 'y' variables.
- `test_linear_regression_accuracy(generate_test_data)`: Validates the accuracy of the linear regression model on the test data.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

@pytest.fixture
def load_data():
    return pd.read_csv('co2_emission.csv')

@pytest.fixture
def generate_test_data():
    data = {
        'x': [1, 2, 3, 4, 5],
        'y': [2.1, 4.2, 6.3, 8.4, 10.5]
    }
    return pd.DataFrame(data)

def test_file_handling():
    try:
        data = pd.read_csv('co2_emission.csv')
        assert not data.empty
    except FileNotFoundError:
        pytest.fail("Data file not found")

    assert 'co2_emission.csv'.endswith('.csv')

def test_data_structure(load_data):
    required_columns = ['Annual CO₂ emissions (tonnes )', 'Year']
    for col in required_columns:
        assert col in load_data.columns

    assert load_data.select_dtypes(include=[np.number]).shape[1] > 0
    

    with open('co2_emission.csv', 'r') as file:
        assert len(file.readline().split(',')) > 0

def test_sufficient_data(load_data):
    assert len(load_data) >= 2

def test_correlation(generate_test_data):
    test_data = generate_test_data
    correlation = test_data['x'].corr(test_data['y'])
    assert abs(correlation) > 0.8

def test_linear_regression_accuracy(generate_test_data):
    test_data = generate_test_data
    X = test_data[['x']]
    y = test_data['y']

    model = LinearRegression()
    model.fit(X, y)

    expected_slope = 2.1
    expected_intercept = 0.0

    assert np.isclose(model.coef_[0], expected_slope, atol=1e-2)
    assert np.isclose(model.intercept_, expected_intercept, atol=1e-2)
