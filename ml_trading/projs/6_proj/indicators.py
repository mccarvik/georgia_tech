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


def calculate_ema(prices, span=20):
    """
    Calculate the Exponential Moving Average (EMA) for the
    """
    ema = prices.ewm(span=span, adjust=False).mean()
    return ema


def calculate_macd(prices, fast_span=12, slow_span=26, signal_span=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) for the given prices
    """
    fast_ema = calculate_ema(prices, span=fast_span)
    slow_ema = calculate_ema(prices, span=slow_span)
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_span, adjust=False).mean()
    return macd, signal


def calculate_cci(prices, high, low, window=20):
    """
    Calculate the Commodity Channel Index (CCI) for the
    """
    tp = (high + low + prices) / 3
    rolling_mean = tp.rolling(window=window).mean()
    rolling_std = tp.rolling(window=window).std()
    cci = (tp - rolling_mean) / (0.015 * rolling_std)
    return cci

