""""""  		  	   		 	 	 			  		 			     			  	 
"""MC1-P2: Optimize a portfolio.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
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
  		  	   		 	 	 			  		 			     			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		 	 	 			  		 			     			  	 
GT User ID: tb34 (replace with your User ID)  		  	   		 	 	 			  		 			     			  	 
GT ID: 900897987 (replace with your GT ID)  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import pdb  		  	   		 	 	 			  		 			     			  	 
import datetime as dt	  	   		 	 	 			  		 			     			  	 	  		 			     			  	 
import numpy as np  		  	   		 	 	 			  		 			     			  	 
from scipy.optimize import minimize  		  	   		 	 	 			  		 			     			  	 
import matplotlib.pyplot as plt  		  	   		 	 	 			  		 			     			  	 
import pandas as pd  		  	   		 	 	 			  		 			     			  	 
from util import get_data, plot_data  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
# This is the function that will be tested by the autograder  		  	   		 	 	 			  		 			     			  	 
# The student must update this code to properly implement the functionality  		  	   		 	 	 			  		 			     			  	 
def optimize_portfolio(  		  	   		 	 	 			  		 			     			  	 
    sd=dt.datetime(2008, 1, 1),  		  	   		 	 	 			  		 			     			  	 
    ed=dt.datetime(2009, 1, 1),  		  	   		 	 	 			  		 			     			  	 
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		 	 	 			  		 			     			  	 
    gen_plot=True,  		  	   		 	 	 			  		 			     			  	 
):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		 	 	 			  		 			     			  	 
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		 	 	 			  		 			     			  	 
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		 	 	 			  		 			     			  	 
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		 	 	 			  		 			     			  	 
    statistics.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 			  		 			     			  	 
    :type sd: datetime  		  	   		 	 	 			  		 			     			  	 
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 			  		 			     			  	 
    :type ed: datetime  		  	   		 	 	 			  		 			     			  	 
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		 	 	 			  		 			     			  	 
        symbol in the data directory)  		  	   		 	 	 			  		 			     			  	 
    :type syms: list  		  	   		 	 	 			  		 			     			  	 
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		 	 	 			  		 			     			  	 
        code with gen_plot = False.  		  	   		 	 	 			  		 			     			  	 
    :type gen_plot: bool  		  	   		 	 	 			  		 			     			  	 
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		 	 	 			  		 			     			  	 
        standard deviation of daily returns, and Sharpe ratio  		  	   		 	 	 			  		 			     			  	 
    :rtype: tuple  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Read in adjusted closing prices for given symbols, date range  		  	   		 	 	 			  		 			     			  	 
    dates = pd.date_range(sd, ed)  		  	   		 	 	 			  		 			     			  	 
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		 	 	 			  		 			     			  	 
    prices = prices_all[syms]  # only portfolio symbols  		  	   		 	 	 			  		 			     			  	 
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # find the allocations for the optimal portfolio  		  	   		 	
    # normalize the prices
    for stock in prices.columns:
        prices[stock] = prices[stock] / prices[stock].iloc[0]
    # print(prices)

    # write sharpe ratio calculator
    # write optimizer function

    # note that the values here ARE NOT meant to be correct for a test case
    # print(prices)	  	   		 	 	 			  		 			     			  	 
    allocs = np.asarray(  		  	   		 	 	 			  		 			     			  	 
        [0.1, 0.1, 0.1, 0.7]  		  	   		 	 	 			  		 			     			  	 
    )  # add code here to find the allocations  		  	   		 	 	 			  		 			     			  	 
    cr, adr, sddr, sr = [  		  	   		 	 	 			  		 			     			  	 
        0.25,  		  	   		 	 	 			  		 			     			  	 
        0.001,  		  	   		 	 	 			  		 			     			  	 
        0.0005,  		  	   		 	 	 			  		 			     			  	 
        2.1,  		  	   		 	 	 			  		 			     			  	 
    ]  # add code here to compute stats

    # print(calc_sharpe(prices, allocs))
    # optimize that portfolio, bruh!!!
    # sharpe_neg = lambda allocs: -1 * calc_sharpe(prices, allocs)
    opt_allocs = port_opt(prices, calc_sharpe)
    print(opt_allocs)	

    # get vals
    allocs = opt_allocs
    sddr = calc_vol(prices, opt_allocs)
    sr = calc_sharpe(prices, opt_allocs)
    cr = calc_cum_ret(prices, opt_allocs)
    adr = calc_avg_daily_ret(prices, opt_allocs)
  		  	   		 	 	 			  		 			     			  	 
    # Get daily portfolio value  		  	   		 	 	 			  		 			     			  	 
    # Compute daily portfolio value
    port_val = (prices * allocs).sum(axis=1)

    # Normalize the portfolio value and SPY
    norm_port_val = port_val / port_val.iloc[0]
    norm_SPY = prices_SPY / prices_SPY.iloc[0]

    # Create a dataframe with both normalized portfolio value and SPY
    df_temp = pd.concat([norm_port_val, norm_SPY], keys=["Portfolio", "SPY"], axis=1)
  		  	   		 	 	 			  		 			     			  	 
    # Compare daily portfolio value with SPY using a normalized plot  		  	   		 	 	 			  		 			     			  	 
    if gen_plot:  		  	   		 	 	 			  		 			     			  	 
        # add code to plot here  		  	   		 	 	 			  		 			     			  	 
        df_temp = pd.concat(  		  	   		 	 	 			  		 			     			  	 
            [port_val, prices_SPY], keys=["Portfolio", "SPY"], axis=1  		  	   		 	 	 			  		 			     			  	 
        )  		  	   		 	 	 			  		 			     	
        df_temp = pd.concat([port_val, prices_SPY], keys=["Portfolio", "SPY"], axis=1)
        df_temp = df_temp / df_temp.iloc[0, :]  # Normalize
        df_temp.plot(title="Daily Portfolio Value and SPY", fontsize=12)
        plt.xlabel("Date")
        plt.ylabel("Normalized Price")
        plt.savefig("plot.png")
  		  	   		 	 	 			  		 			     			  	 
    return allocs, cr, adr, sddr, sr  		  	   		 	 	 			  		 			     			  	 


def calc_vol(port_df, alloc):
    """
    Calculate the volatility
    """
    # Calculate daily returns
    daily_returns = port_df.pct_change().dropna()
    # Calculate portfolio daily returns
    port_daily_returns = (daily_returns * alloc).sum(axis=1)
    # Calculate statistics
    std_daily_return = port_daily_returns.std()
    return std_daily_return


def calc_cum_ret(port_df, alloc):
    """
    Calculate the cumulative return
    """
    # Calculate daily returns
    daily_returns = port_df.pct_change().dropna()
    # Calculate portfolio daily returns
    port_daily_returns = (daily_returns * alloc).sum(axis=1)
    # Calculate statistics
    cum_ret = port_daily_returns[-1] / port_daily_returns[0] - 1
    return cum_ret


def calc_avg_daily_ret(port_df, alloc):
    """
    Calculate the average daily return
    """
    # Calculate daily returns
    daily_returns = port_df.pct_change().dropna()
    # Calculate portfolio daily returns
    port_daily_returns = (daily_returns * alloc).sum(axis=1)
    # Calculate statistics
    avg_daily_return = port_daily_returns.mean()
    return avg_daily_return


def calc_sharpe(port_df, alloc):
    """
    Calculate the Sharpe ratio
    """
    # risk free rate = 0, this assumption is almost certainly not true
    
    # apply allocations
    alloc_port = port_df * alloc

    # Calculate position values
    position_values = alloc_port.sum(axis=1)

    # get dauily returns
    daily_rets = position_values.pct_change().dropna()

    # Calculate statistics
    mean_daily_return = daily_rets.mean()
    std_daily_return = daily_rets.std()

    sharpe_ratio = np.sqrt(252) * mean_daily_return / std_daily_return
    return sharpe_ratio
    # Needs to be negative for min function!!!
    # return sharpe_ratio


def port_opt(prices, func):
    """
    Optimize the portfolio
    """
    # Number of assets
    num_assets = len(prices.columns)
    
    # Initial guess (equal allocation) - as good a place as any to start
    init_guess = np.array([1.0 / num_assets] * num_assets)
    
    # Constraints: allocations must sum to 1 as all port must be allocated to one of these stocks
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: allocations must be between 0 and 1
    # for this assignment we wont allow shorting
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Minimize the negative Sharpe ratio
    # result = minimize(lambda allocs: -func(prices, allocs), init_guess, method='SLSQP', bounds=bounds, constraints=cons).x
    # Minimize the negative of the objective function
    result = minimize(
        fun=lambda allocs: -func(prices, allocs),
        x0=init_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'ftol': 1e-9, 'disp': False}
    ).x
    return result


def str2dt(strng):  		  	   		 	 	 			  		 			     			  	 
    year, month, day = map(int, strng.split("-"))  		  	   		 	 	 			  		 			     			  	 
    return dt.datetime(year, month, day)  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 	
	 	 	 			  		 			     			  	 
def test_code():  		  	   		 	 	 			  		 			     			  	 

    """  		  	   		 	 	 			  		 			     			  	 
    This function WILL NOT be called by the auto grader.  		  	   		 	 	 			  		 			     			  	 
    """
    # start_date=str2dt("2008-01-01")  		  	   		 	 	 			  		 			     			  	 
    # end_date=str2dt("2009-01-01") 		  	   		 	 	 			  		 			     			  	 
    # symbols=["GOOG", "AAPL", "GLD", "XOM"]
  		  	   		 	 	 			  		 			     			  	 
    # start_date=str2dt("2010-01-01")  		  	   		 	 	 			  		 			     			  	 
    # end_date=str2dt("2010-12-31") 		  	   		 	 	 			  		 			     			  	 
    # symbols=["GOOG", "AAPL", "GLD", "XOM"]

    # start_date=str2dt("2004-01-01")	  	   		 	 	 			  		 			     			  	 
    # end_date=str2dt("2006-01-01")	  	   		 	 	 			  		 			     			  	 
    # symbols=["AXP", "HPQ", "IBM", "HNZ"]

    # start_date=str2dt("2010-01-01") 	 
    # end_date=str2dt("2010-12-31")
    # symbols=["AXP", "HPQ", "IBM", "HNZ"]

    # start_date=str2dt("2004-12-01")
    # end_date=str2dt("2006-05-31")
    # symbols=["YHOO", "XOM", "GLD", "HNZ"]

    # start_date = dt.datetime(2009, 1, 1)  		  	   		 	 	 			  		 			     			  	 
    # end_date = dt.datetime(2010, 1, 1)  		  	   		 	 	 			  		 			     			  	 
    # symbols = ["GOOG", "AAPL", "GLD", "XOM", "IBM"]  

    start_date=str2dt("2008-06-01")	  	   		 	 	 			  		 			     			  	 
    end_date=str2dt("2009-06-01")
    symbols = ["IBM", "X", "GLD", "JPM"]  	  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Assess the portfolio  		  	   		 	 	 			  		 			     			  	 
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		 	 	 			  		 			     			  	 
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True  		  	   		 	 	 			  		 			     			  	 
    )  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Print statistics  		  	   		 	 	 			  		 			     			  	 
    print(f"Start Date: {start_date}")  		  	   		 	 	 			  		 			     			  	 
    print(f"End Date: {end_date}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Symbols: {symbols}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Allocations:{allocations}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Sharpe Ratio: {sr}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Average Daily Return: {adr}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Cumulative Return: {cr}")  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    # This code WILL NOT be called by the auto grader  		  	   		 	 	 			  		 			     			  	 
    # Do not assume that it will be called  		  	   		 	 	 			  		 			     			  	 
    test_code()  		  	   		 	 	 			  		 			     			  	 
