"""
This is a test project file.
"""

import pandas as pd
import TheoreticallyOptimalStrategy as tos
import marketsimcode as marksim
import util
import matplotlib.pyplot as plt
import datetime as dt
import pdb
import indicators as ind
from matplotlib.gridspec import GridSpec


def author():
    """
    :return: The GT username of the student
    """
    return 'kmccarville3'

def plot_indicator(df, ind_name):
    """
    function to plot the indicator
    """
    import matplotlib.dates as mdates
    fig = plt.figure()
    plt.plot(df)
    plt.title(ind_name)
    plt.xlabel('Date')
    plt.ylabel('Indicator Value / Price')
    plt.legend(df.columns)
    
    # format x-axis to be every 3 months
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.savefig(ind_name + '.png')
    plt.close()


def plot_rsi(df):
    """
    function to plot the rsi and price in a single figure with two subplots
    """
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, height_ratios=[1, 2], hspace=0.1)
    
    # RSI subplot (top)
    ax1 = plt.subplot(gs[0])
    ax1.plot(df.index, df['RSI'], color='blue', linewidth=1.5)
    # ax1.axhline(75, color='red', linestyle='--', alpha=0.5)
    # ax1.axhline(25, color='green', linestyle='--', alpha=0.5)
    ax1.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax1.axhline(50, color='gray', linestyle='--', alpha=0.3)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('RSI (21)', fontsize=12)
    ax1.set_title(f'JPM Relative Strength Index (RSI)', fontsize=14)
    ax1.grid(alpha=0.3)
    
    # Stock price subplot (bottom)
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(df.index, df['JPM'], color='black', linewidth=1.5)
    ax2.set_ylabel('Price ($)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title(f'JPM Stock Price', fontsize=14)
    ax2.grid(alpha=0.3)
    
    # Format x-axis
    plt.setp(ax1.get_xticklabels(), visible=False)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig('RSI.png')
    plt.close()


def plot_bollinger_bands(df):
    """
    function to plot the bollinger bands and price in a single figure with two subplots
    """
    window = 21
    rolling_mean = df['JPM'].rolling(window=window).mean()
    rolling_std = df['JPM'].rolling(window=window).std()
    upperbound = rolling_mean + (rolling_std * 2)
    lowerbound = rolling_mean - (rolling_std * 2)

    fig = plt.figure(figsize=(12, 8))

    # Bollinger Bands subplot (top)
    plt.plot(df.index, df['JPM'], color='black', linewidth=1.5, label='Price')
    plt.plot(df.index, upperbound, color='red', linestyle='--', linewidth=1, label='Upper Band')
    plt.plot(df.index, lowerbound, color='green', linestyle='--', linewidth=1, label='Lower Band')
    plt.plot(df.index, rolling_mean, color='blue', linestyle='--', linewidth=1, label='Mean')
    plt.fill_between(df.index, lowerbound, upperbound, color='gray', alpha=0.2)
    plt.ylabel('Price ($)', fontsize=12)
    plt.title(f'JPM Bollinger Bands', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)

    # Format x-axis
    plt.tight_layout()
    plt.savefig('Bollinger_Bands.png')
    plt.close()


def plot_macd(df):
    """
    function to plot the macd and price in a single figure with two subplots
    """
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Create figure with GridSpec
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.15)
    
    # Price subplot (top)
    ax1 = plt.subplot(gs[0])
    ax1.plot(df.index, df['JPM'], color='blue', linewidth=1.5, label='Close Price')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'JPM Stock Price', fontsize=14)
    ax1.grid(alpha=0.3)
    ax1.legend(loc='best')
    
    # MACD subplot (bottom)
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(df.index, df['macd'], color='blue', linewidth=1.5, label='MACD Line')
    ax2.plot(df.index, df['macd_signal'], color='red', linewidth=1, label='Signal Line')
    
    # # Plot histogram as bar chart
    # ax2.bar(df.index, df['macd_hist'], color=df['macd_hist'].apply(
    #     lambda x: 'green' if x > 0 else 'red'), alpha=0.5, label='Histogram')
    
    # Add horizontal line at 0
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    ax2.set_ylabel('MACD', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title(f'MACD (12, 26, 9)', fontsize=14)
    ax2.grid(alpha=0.3)
    ax2.legend(loc='best')
    
    # Format x-axis
    plt.setp(ax1.get_xticklabels(), visible=False)
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig('MACD.png')
    plt.close()



def plot_cci(df, cci_window):
    """
    function to plot the cci and price in a single figure with two subplots
    """
    # Create figure with GridSpec
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.15)
    
    # Price subplot (top)
    ax1 = plt.subplot(gs[0])
    ax1.plot(df.index, df['Close'], color='blue', linewidth=1.5)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'JPM Stock Price', fontsize=14)
    ax1.grid(alpha=0.3)
    
    # CCI subplot (bottom)
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(df.index, df['cci'], color='purple', linewidth=1.5)
    
    # Add horizontal lines at +100, 0, -100 (common CCI reference levels)
    ax2.axhline(100, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.axhline(-100, color='green', linestyle='--', alpha=0.5)
    
    ax2.set_ylabel(f'CCI ({cci_window})', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title(f'Commodity Channel Index ({cci_window}-period)', fontsize=14)
    ax2.grid(alpha=0.3)
    
    # Format x-axis
    plt.setp(ax1.get_xticklabels(), visible=False)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig('CCI.png')
    plt.close()


if __name__ == "__main__":
    # grab dates
    sd = dt.date(2008, 1, 1)
    ed = dt.date(2009, 12, 31)
    date_ranges = pd.date_range(sd, ed)
    symbols = ["JPM"]  # must be a singular list element

    # get trades
    trades = tos.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=100000)
    orders = pd.DataFrame(index=trades.index.values, columns=["Symbol", "Order", "Shares"])
    orders["Symbol"] = "JPM"
    orders["Order"] = trades.where(trades > 0, "BUY").where(trades < 0, "SELL")
    orders["Shares"] = abs(trades)

    port_val = marksim.compute_portvals(orders)
    # get it in percentage terms
    port_val = port_val / port_val.iloc[0]
    jpm = 1000 * util.get_data(["JPM"], pd.date_range(sd, ed), addSPY=False, colname="Adj Close").dropna()
    jpm['bench'] = jpm / jpm.iloc[0, 0]
    jpm['jpm_perc'] = port_val


    # plot
    fig = plt.figure()
    plt.plot(jpm['jpm_perc'], label='Optimal Strategy')
    plt.plot(jpm['bench'], label='Benchmark')
    plt.title('Theoretically Optimal Strategy vs Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.legend()
    import matplotlib.dates as mdates
    
    # format x-axis to be by month
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.savefig('Theoretically Optimal Strategy vs Benchmark.png')
    plt.close()


    # Now time to do the indicators
    # first we have to get te data
    px = util.get_data(symbols, date_ranges, addSPY=False, colname="Adj Close").dropna()
    low = util.get_data(symbols, date_ranges, addSPY=False, colname="Low").dropna()
    high = util.get_data(symbols, date_ranges, addSPY=False, colname="High").dropna()
    volume = util.get_data(symbols, date_ranges, addSPY=False, colname="Volume").dropna()

    # ema
    ema = ind.calculate_ema(px)
    df = pd.DataFrame({'JPM': px['JPM'], 'EMA': ema['JPM']})
    plot_indicator(df, 'Exponential Moving Average')

    # rsi
    rsi = ind.calculate_rsi(px)
    df = pd.DataFrame({'JPM': px['JPM'], 'RSI': rsi['JPM']})
    plot_rsi(df)

    # bollinger bands
    bb = ind.calculate_bollinger_bands(px)
    df = pd.DataFrame({'JPM': px['JPM'], 'bb': bb['JPM']})
    plot_bollinger_bands(df)

    # cci
    cci = ind.calculate_cci(px, high, low)
    df = pd.DataFrame({'Close': px['JPM'], 'cci': cci['JPM']})
    plot_cci(df, 21)

    # macd
    macd = ind.calculate_macd(px)
    df = pd.DataFrame({'JPM': px['JPM'], 'macd': macd['JPM']})
    plot_macd(df)





