import pytest
import pandas as pd

def test_data_columns():
    # Check if the dataset has the expected columns
    df = pd.read_csv('data.csv')
    assert 'Year' in df.columns
    assert 'CO2_emission' in df.columns

def test_positive_emissions():
    # Check if CO2 emissions are all positive
    df = pd.read_csv('data.csv')
    assert (df['CO2_emission'] >= 0).all()
