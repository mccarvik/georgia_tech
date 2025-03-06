import pandas as pd

"""
This module contains the functions to calculate the technical indicators.
"""

def calculate_bollinger_bands(prices, window=20):
    """
    Calculate Bollinger Bands.
    """
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return rolling_mean, upper_band, lower_band

