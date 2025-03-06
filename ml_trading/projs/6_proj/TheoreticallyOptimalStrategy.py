"""
Theoretically Optimal Strategy
"""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data

def author():
    """
    :return: The GT username of the student
    """
    return 'kmccarville3'

def testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):
    """
    :param symbol: The stock symbol to act on
    :param sd: A datetime object that represents the start date
    :param ed: A datetime object that represents the end date
    :param sv: Start value of the portfolio
    """ 

    # load data from get_data
    data = get_data([symbol], pd.date_range(sd, ed), addSPY=False, colname="Adj Close")

    
    # get rets
    rets = data.diff()

    # decide whether to buy or sell based on optimal information
    trade_signal = rets.where(rets > 0, "BUY").where(rets < 0, "SELL")
    # and then shift it back one day early as we know what to do based on looking in the future
    trade_signal = trade_signal.shift(-1)

    # create trades dataframe
    trades = pd.DataFrame(data=0.0, columns=["Shares"], index=trade_signal.index.values)

    # revert the position the next day. This might mean we buy and sell the same day but that
    # okay, there are assumed to be no transaction costs
    for i in range(trade_signal.shape[0]):
        if trade_signal.iloc[i, 0] == "BUY":
            trades.iloc[i, 0] = 1000
        else:
            trades.iloc[i, 0] = -1000
    
    return trades