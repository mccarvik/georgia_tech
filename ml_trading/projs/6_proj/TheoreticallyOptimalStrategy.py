"""
Theoretically Optimal Strategy
"""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data
import pdb

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
    # adding SPY to fix dates
    data = get_data([symbol], pd.date_range(sd, ed), addSPY=True, colname="Adj Close")

    
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
    # pdb.set_trace()
    trade_signal = trade_signal.drop(columns="SPY")

    pos_shares = True

    for i in range(trade_signal.shape[0]):
        # need this to determine the direction of the trade and if we go 2000 shares
        # these are trades NOT position
        direction = 0
        if trade_signal.iloc[i, 0] == "BUY":
            direction = 1
        else:
            direction = -1

        if i == 0:
            trades['Shares'].iloc[i] = 1000 * direction
            if not direction == 1:
                pos_shares = False
        else:
            if (pos_shares and direction > 0) or (not pos_shares and direction < 0):
                trades['Shares'].iloc[i] = 0
            else:
                trades['Shares'].iloc[i] = 2000 * direction
                pos_shares = not pos_shares
        
    return trades