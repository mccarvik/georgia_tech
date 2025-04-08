""""""  		  	   		 	 	 			  		 			     			  	 
"""MC2-P1: Market simulator.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			     			  	 
All Rights Reserved  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			     			  	 
or edited.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			     			  	 
GT honor code violation.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Student Name: Kevin McCarville (replace with your name)  		  	   		 	 	 			  		 			     			  	 
GT User ID: kmccarville3 (replace with your User ID)  		  	   		 	 	 			  		 			     			  	 
GT ID: 903969483 (replace with your GT ID)  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 

import pdb	   		 	 	 			  		 			     			  	 
import datetime as dt  		  	   		 	 	 			  		 			     			  	 
import os	  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import numpy as np  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import pandas as pd  		  	   		 	 	 			  		 			     			  	 
from util import get_data, plot_data  		  	   		 	 	 			  		 			     			  	 

def author():
    """
    :return: The username of the student
    """
    return 'kmccarville3'	  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
def compute_portvals(  		  	   		 	 	 			  		 			     			  	 
    orders,		  	   		 	 	 			  		 			     			  	 
    start_val=100000,  		  	   		 	 	 			  		 			     			  	 
    commission=0,  		  	   		 	 	 			  		 			     			  	 
    impact=0.000,  		  	   		 	 	 			  		 			     			  	 
):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    Computes the portfolio values.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param orders_file: Path of the order file or the file object  		  	   		 	 	 			  		 			     			  	 
    :type orders_file: str or file object  		  	   		 	 	 			  		 			     			  	 
    :param start_val: The starting value of the portfolio  		  	   		 	 	 			  		 			     			  	 
    :type start_val: int  		  	   		 	 	 			  		 			     			  	 
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		 	 	 			  		 			     			  	 
    :type commission: float  		  	   		 	 	 			  		 			     			  	 
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		 	 	 			  		 			     			  	 
    :type impact: float  		  	   		 	 	 			  		 			     			  	 
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		 	 	 			  		 			     			  	 
    :rtype: pandas.DataFrame  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    # this is the function the autograder will call to test your code  		  	   		 	 	 			  		 			     			  	 
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		 	 	 			  		 			     			  	 
    # code should work correctly with either input  		  	   		 	 	 			  		 			     			  	 
    # TODO: Your code here

    # grab orders
    # mkt_orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    # mkt_orders = pd.read_csv(orders, index_col='Date', header=0)
    mkt_orders = orders
  		  	   		 	 	 			  		 			     			  	 
    # In the template, instead of computing the value of the portfolio, we just  		  	   		 	 	 			  		 			     			  	 
    # # read in the value of IBM over 6 months  		  	   		 	 	 			  		 			     			  	 
    # start_date = dt.datetime(2008, 1, 1)  		  	   		 	 	 			  		 			     			  	 
    # end_date = dt.datetime(2008, 6, 1)  		  	   		 	 	 			  		 			     			  	 
    # portvals = get_data(["IBM"], pd.date_range(start_date, end_date))  		  	   		 	 	 			  		 			     			  	 
    # portvals = portvals[["IBM"]]  # remove SPY  		  	   		 	 	 			  		 			     			  	 
    # rv = pd.DataFrame(index=portvals.index, data=portvals.values)  		  	   		 	 	 			  		 			     			  	 

    # grab dates
    start = pd.to_datetime(orders.index.min())
    end = pd.to_datetime(orders.index.max())

    # grab stocks
    stocks = list(set(orders['Symbol'].values))

    # grab data
    data = get_data(stocks, pd.date_range(start, end))
    data.drop(columns=['SPY'], inplace=True)
    data['Cash'] = 1.0

    # create trades dataframe
    trades = pd.DataFrame(data=0.0, index=data.index, columns=stocks + ['Cash'])

    # fill trades dataframe
    for i in range(len(mkt_orders)):
        date = pd.to_datetime(mkt_orders.index[i]).date()
        date = pd.Timestamp(date)
        stock = mkt_orders['Symbol'][i]
        shares = mkt_orders['Shares'][i]
        order = mkt_orders['Order'][i]
        try:
            if order == 'BUY':
                trades.loc[date, stock] += shares
                trades.loc[date, 'Cash'] -= (data.loc[date, stock] * shares) + commission
            elif order == 'SELL':
                trades.loc[date, stock] -= shares
                trades.loc[date, 'Cash'] += (data.loc[date, stock] * shares) - commission
            # pdb.set_trace()
            day_price = data.loc[date, stock]
            imp = shares * day_price * impact
            trades.loc[date, 'Cash'] -= imp
        except KeyError as exc:
            print(exc)
            continue


    
    # create holdings dataframe
    # holdings = trades.copy()
    pdb.set_trace()
    holdings = pd.DataFrame(data=0.0, index=data.index, columns=stocks + ['Cash'])

    # holdings['Cash'] = start_val
    holdings.iloc[0] = trades.iloc[0]
    holdings['Cash'].iat[0] += float(start_val)
    # holdings = holdings.cumsum()

    for i in range(1, holdings.shape[0]):
        # take the previous day's holdings and multiply by the previous day's prices
        holdings.iloc[i] = holdings.iloc[i-1] + trades.iloc[i]

    # create values dataframe
    values = holdings * data
    
    # create portvals dataframe
    portvals = values.sum(axis=1)
    portvals = pd.DataFrame(portvals, columns=['Portfolio Value'])
    pdb.set_trace()
    return portvals
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
def test_code():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    Helper function to test code  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    # this is a helper function you can use to test your code  		  	   		 	 	 			  		 			     			  	 
    # note that during autograding his function will not be called.  		  	   		 	 	 			  		 			     			  	 
    # Define input parameters  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    of = "./orders/orders-12.csv"  		  	   		 	 	 			  		 			     			  	 
    sv = 1000000
    commission = 0  
    impact = 0.005		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Process orders  		  	   		 	 	 			  		 			     			  	 
    portvals = compute_portvals(orders_file=of, start_val=sv, commission=commission, impact=impact)  		  	   		 	 	 			  		 			     			  	 
    if isinstance(portvals, pd.DataFrame):  		  	   		 	 	 			  		 			     			  	 
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		 	 	 			  		 			     			  	 
    else:  		  	   		 	 	 			  		 			     			  	 
        "warning, code did not return a DataFrame"  
    print(portvals)		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Get portfolio stats  		  	   		 	 	 			  		 			     			  	 
    # Here we just fake the data. you should use your code from previous assignments.  		  	   		 	 	 			  		 			     			  	 
    start_date = dt.datetime(2008, 1, 1)  		  	   		 	 	 			  		 			     			  	 
    end_date = dt.datetime(2008, 6, 1)  		  	   		 	 	 			  		 			     			  	 
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [  		  	   		 	 	 			  		 			     			  	 
        0.2,  		  	   		 	 	 			  		 			     			  	 
        0.01,  		  	   		 	 	 			  		 			     			  	 
        0.02,  		  	   		 	 	 			  		 			     			  	 
        1.5,  		  	   		 	 	 			  		 			     			  	 
    ]  		  	   		 	 	 			  		 			     			  	 
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [  		  	   		 	 	 			  		 			     			  	 
        0.2,  		  	   		 	 	 			  		 			     			  	 
        0.01,  		  	   		 	 	 			  		 			     			  	 
        0.02,  		  	   		 	 	 			  		 			     			  	 
        1.5,  		  	   		 	 	 			  		 			     			  	 
    ]  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Compare portfolio against $SPX  		  	   		 	 	 			  		 			     			  	 
    print(f"Date Range: {start_date} to {end_date}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    test_code()  		  	   		 	 	 			  		 			     			  	 
