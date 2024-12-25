import pytest
import pandas as pd

def test_data_columns():
    # Check if the dataset has the expected columns
    df = pd.read_csv('co2_emission.csv')
    assert 'Year' in df.columns
    assert 'Annual CO₂ emissions (tonnes )' in df.columns

