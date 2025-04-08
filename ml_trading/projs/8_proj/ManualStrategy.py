"""
Implementing ManualStrategy class
"""

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import indicators as ind
from util import get_data
import pdb


def author():
    """
    @return: The name of the author
    """
    return "kmccarville3"


def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    """
    function to implement a manual trading strategy
    """

    # given parameters
    max_position = 1000 # max number of shares to hold

    # set manual parameters
    look_back = 21 # number of days to look back, 21 trading days in a month

    # get stock price data
    dates = pd.date_range(sd, ed)
    dfStockPrice = get_data([symbol], dates, True, colname='Adj Close').drop(columns=['SPY'])
    dfStockPrice.sort_index()
    dfStockPrice = dfStockPrice.ffill().bfill()
    dfStockPriceNorm = dfStockPrice / dfStockPrice.iloc[0, :]
    dates = dfStockPriceNorm.index

    # set up orders dataframe
    orders = pd.DataFrame(0, index=dates, columns=['order type', 'position', symbol])
    rsi = ind.calculate_rsi(dfStockPriceNorm, look_back)
    bboll = ind.calculate_bollinger_bands(dfStockPriceNorm, look_back)
    bboll = reconstruct_bollinger_bands(bboll, dfStockPriceNorm, look_back)
    ema = ind.calculate_ema(dfStockPriceNorm, look_back)
    ema = ema_signal(ema, dfStockPriceNorm, symbol)

    current_holdings = 0
    i = 1
    # Manual trading strategy

    for index, row in dfStockPriceNorm.iterrows():
        # Check if we are at the end of the dataframe
        if i >= len(dfStockPriceNorm):
            break

        # Convert index to datetime if not already
        index = pd.to_datetime(index)
        
        # get signals
        rsi_val = rsi.loc[index]['JPM']
        bboll_val = bboll[i]
        ema_val = ema[i]

        # need to have enough lookback first
        if i > look_back:
            # check for buy signal
            rsi_sig = False
            if rsi_val < 30:
                rsi_sig = True
            
            bboll_sig = False
            if bboll_val[0] < -2:
                bboll_sig = True
            
            ema_sig = False
            if ema_val == 1:
                ema_sig = True

            if rsi_sig or bboll_sig or ema_sig and current_holdings < max_position:
                # buy signal
                orders.loc[index, 'order type'] = 1
                if current_holdings == 0:
                    orders.loc[index, 'position'] = 1000
                    orders.loc[index, symbol] = 1000
                else:
                    orders.loc[index, 'position'] = 2000
                    orders.loc[index, symbol] = 2000
                current_holdings = 1000
            
            # check for sell signal
            rsi_sig = False
            if rsi_val > 70:
                rsi_sig = True

            bboll_sig = False
            if bboll_val[1] > 2:
                bboll_sig = True

            ema_sig = False
            if ema_val == -1:
                ema_sig = True
            
            if rsi_sig or bboll_sig or ema_sig and current_holdings > -max_position:
                # sell signal
                orders.loc[index, 'order type'] = -1
                if current_holdings == 0:
                    orders.loc[index, 'position'] = 1000
                    orders.loc[index, symbol] = -1000
                else:
                    orders.loc[index, 'position'] = 2000
                    orders.loc[index, symbol] = -2000
                current_holdings = -1000
            
        # increment index
        i += 1

    # set the order type
    return orders


def ema_signal(ema, prices, symbol):
    """
    Function to generate buy/sell signals based on the EMA.
    A buy signal (1) is generated when the price crosses above the EMA,
    and a sell signal (-1) is generated when the price crosses below the EMA.
    """
    signals = []
    prev_price = None
    prev_ema = None

    for date in prices.index:
        current_price = prices.loc[date, symbol]
        current_ema = ema.loc[date, symbol]

        if prev_price is not None and prev_ema is not None:
            if prev_price <= prev_ema and current_price > current_ema:
                signals.append(1)  # Buy signal
            elif prev_price >= prev_ema and current_price < current_ema:
                signals.append(-1)  # Sell signal
            else:
                signals.append(0)  # No signal
        else:
            signals.append(0)  # No signal for the first data point

        prev_price = current_price
        prev_ema = current_ema

    return signals


def gen_plot(man_strat, bench, symbol, trades, in_sample=True):
    """
    Function to plot the manual strategy vs benchmark strategy
    """
    alpha = 0.7
    fig, ax = plt.subplots(figsize=(8, 6))

    man_strat = man_strat / man_strat.iloc[0]
    bench = bench / bench.iloc[0]
    ax.plot(man_strat, color='red')
    ax.plot(bench, color='purple')
    ax.grid()
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend(['Manual', "Benchmark"])

    # plot trades
    for index, row in trades.iterrows():
        if trades.loc[index][symbol] > 0:
            ax.axvline(x=index, color='blue', linestyle='--', alpha=alpha)
        elif trades.loc[index][symbol] < 0:
            ax.axvline(x=index, color='black', linestyle='--', alpha=alpha)
    
    if in_sample:
        plt.title("In Sample Manual Strategy vs Benchmark - {}".format(symbol))
        plt.savefig("ManualStrategy_{}_in_sample.png".format(symbol))
    else:
        plt.title("Out of Sample Manual Strategy vs Benchmark - {}".format(symbol))
        plt.savefig("ManualStrategy_{}_out_of_sample.png".format(symbol))


def bench_rets(symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    """
    Function to get the benchmark returns
    """
    # get stock price data
    dates = pd.date_range(sd, ed)
    dfStockPrice = get_data([symbol], dates, True, colname='Adj Close').drop(columns=['SPY'])
    dfStockPrice.sort_index()
    dfStockPrice = dfStockPrice.ffill().bfill()
    dfStockPriceNorm = dfStockPrice / dfStockPrice.iloc[0, :]
    return dfStockPriceNorm


def cum_ret(dataframe):
    """
    Function to calculate the cumulative returns of a dataframe
    """
    # Calculate daily returns
    # Convert numpy array to DataFrame
    dataframe = pd.DataFrame(dataframe)
    
    # Calculate daily returns
    daily_returns = dataframe.pct_change().fillna(0)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    return cumulative_returns


def daily_ret(dataframe):
    """
    Function to calculate the daily returns of a dataframe
    """
    # Calculate daily returns
    daily_returns = dataframe.pct_change().fillna(0)
    return daily_returns


def reconstruct_bollinger_bands(boll, prices, window=21):
    """
    Reconstruct Bollinger Bands from the normalized boll value.
    
    Parameters:
    - boll: Normalized Bollinger Bands series from original function
    - prices: Original price series
    - window: Rolling window size (default 21)
    
    Returns:
    - DataFrame with original prices, middle band, upper band, and lower band
    """
    # Calculate rolling mean and standard deviation
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    
    # Reconstruct Bollinger Bands
    # boll = (prices - rolling_mean) / (2 * rolling_std)
    # So, prices = boll * (2 * rolling_std) + rolling_mean
    
    # Middle Band (Simple Moving Average)
    middle_band = rolling_mean
    
    # Upper and Lower Bands (2 standard deviations from the mean)
    upper_band = rolling_mean + (boll * (2 * rolling_std))
    lower_band = rolling_mean - (boll * (2 * rolling_std))
    
    # Combine into a list of tuples where each tuple contains lower and upper band values
    bollinger_bands = list(zip(lower_band.squeeze(), upper_band.squeeze()))
    return bollinger_bands
    

def print_strategy_returns(man_strat, bench):
    """
    Function to print the cumulative, daily returns, and standard deviation of both strategies.
    
    Parameters:
    - man_strat: DataFrame of manual strategy portfolio values
    - bench: DataFrame of benchmark strategy portfolio values
    """
    # Calculate cumulative returns
    man_cum_ret = cum_ret(man_strat.values).iloc[-1]
    bench_cum_ret = cum_ret(bench.values).iloc[-1]
    
    # Calculate daily returns
    man_daily_ret = daily_ret(man_strat)
    bench_daily_ret = daily_ret(bench)
    
    # Calculate standard deviation of daily returns
    man_std_dev = man_daily_ret.std()
    bench_std_dev = bench_daily_ret.std()
    
    # Print results
    print("Manual Strategy Returns:")
    print(f"Cumulative Return: " + str(man_cum_ret[0]))
    print(f"Average Daily Return: " + str(man_daily_ret.mean()))
    print(f"Standard Deviation of Daily Returns: " + str(man_std_dev))

    print("Benchmark Strategy Returns:")
    print(f"Cumulative Return: " + str(bench_cum_ret[0]))
    print(f"Average Daily Return: " + str(bench_daily_ret.mean()))
    print(f"Standard Deviation of Daily Returns: " + str(bench_std_dev))


def plot_manual_vs_benchmark(symbol, sd, ed, sv=100000, in_sample=True, commission=0.0, impact=0.0):
    """
    Generate plots comparing the manual strategy vs benchmark strategy.

    Parameters:
    - symbol: Stock symbol
    - sd: Start date
    - ed: End date
    - sv: Starting value of the portfolio
    """
    # Generate manual strategy orders
    manual_orders = testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    
    # Calculate manual strategy portfolio values
    prices = get_data([symbol], pd.date_range(sd, ed), True, colname='Adj Close')[symbol]
    # Calculate portfolio values based on manual orders
    manual_portfolio = pd.Series(index=prices.index, dtype=float)
    holdings = 0
    cash = sv

    for date in prices.index:
        if date in manual_orders.index:
            order = manual_orders.loc[date]
            if order['order type'] == 1:  # Buy
                if holdings == 1000:
                    continue
                # pdb.set_trace()
                shares = order['position']
                cost = shares * prices.loc[date]
                imp = shares * prices.loc[date] * impact
                cash -= cost
                cash -= commission + imp
                holdings = 1000
                print(" buy {} shares at {}".format(shares, prices.loc[date]))
            elif order['order type'] == -1:  # Sell
                if holdings == -1000:
                    continue
                # pdb.set_trace()
                shares = order['position']
                revenue = shares * prices.loc[date]
                imp = shares * prices.loc[date] * impact
                cash += revenue
                cash -= commission + imp
                holdings = -1000
                print(" sell {} shares at {}".format(shares, prices.loc[date]))

        # Calculate portfolio value for the day
        manual_portfolio[date] = cash + (holdings * prices.loc[date])
    # manual_portfolio = manual_orders[symbol].cumsum() * get_data([symbol], pd.date_range(sd, ed), True, colname='Adj Close')[symbol]
    
    # Generate benchmark portfolio values
    benchmark = bench_rets(symbol, sd, ed, sv)
    benchmark_portfolio = benchmark * sv
    
    # Plot the results
    gen_plot(manual_portfolio, benchmark_portfolio, symbol, manual_orders)
    print_strategy_returns(manual_portfolio, benchmark_portfolio)
    return manual_portfolio, benchmark_portfolio


if __name__ == "__main__":
    # Define the stock symbol and date range
    symbol = "JPM"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2011, 12, 31)

    # Generate plots for in-sample data
    plot_manual_vs_benchmark(symbol, start_date, end_date, in_sample=True, commission=9.95, impact = 0.005)

    # Generate plots for out-of-sample data
    out_of_sample_start = dt.datetime(2012, 1, 1)
    out_of_sample_end = dt.datetime(2013, 12, 31)
    plot_manual_vs_benchmark(symbol, out_of_sample_start, out_of_sample_end, in_sample=False, commission=9.95, impact=0.005)