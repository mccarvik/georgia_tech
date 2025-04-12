"""
Experiment 1: Compare the performance of a manual strategy with a strategy learner.
"""

import ManualStrategy as ms
import StrategyLearner as sl
import matplotlib.pyplot as plt
import marketsimcode
import datetime as dt
import pdb
import pandas as pd
import numpy as np

def author():
    """
    Returns the author's name as a string.
    """
    return 'kmccarville3'


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


def compare(stock, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000, commission=9.95, impact=0.005):
    """
    Compare the performance of a manual strategy with a strategy learner.
    """
    # impact=0.02
    manual, benchmark = ms.plot_manual_vs_benchmark(stock, sd, ed, sv, True, commission, impact)
    learner = sl.StrategyLearner(impact=impact, commission=commission)
    learner.add_evidence(stock, sd, ed, sv)
    strat = learner.testPolicy(stock, sd, ed, sv)
    strat['Symbol']= stock
    strat['Order'] = strat['Shares'].apply(lambda x: 'BUY' if x > 0 else ('SELL' if x < 0 else 'HOLD'))
    strat.columns = ['Shares', 'Symbol', 'Order']
    strategy = marketsimcode.compute_portvals(strat, sv, commission, impact)
    generate_plot(stock, manual, benchmark, strategy, in_sample=True)
    print_strategy_returns(manual, benchmark, strategy)

    # out-of-sample
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    manual, benchmark = ms.plot_manual_vs_benchmark(stock, sd, ed, sv, True, commission, impact)
    # learner = sl.StrategyLearner(impact=impact, commission=commission)
    # learner.add_evidence(stock, sd, ed, sv)
    strat = learner.testPolicy(stock, sd, ed, sv)
    strat['Symbol']= stock
    strat['Order'] = strat['Shares'].apply(lambda x: 'BUY' if x > 0 else ('SELL' if x < 0 else 'HOLD'))
    strat.columns = ['Shares', 'Symbol', 'Order']
    strategy = marketsimcode.compute_portvals(strat, sv, commission, impact)
    generate_plot(stock, manual, benchmark, strategy, in_sample=False)
    print_strategy_returns(manual, benchmark, strategy)


def print_strategy_returns(man_strat, bench, learn_strat):
    """
    Function to print the cumulative, daily returns, and standard deviation of all strategies.
    
    Parameters:
    - man_strat: DataFrame of manual strategy portfolio values
    - bench: DataFrame of benchmark strategy portfolio values
    - learn_strat: DataFrame of strategy learner portfolio values
    """
    # Calculate cumulative returns
    man_cum_ret = cum_ret(man_strat.values).iloc[-1]
    bench_cum_ret = cum_ret(bench.values).iloc[-1]
    learn_cum_ret = cum_ret(learn_strat.values).iloc[-1]
    
    # Calculate daily returns
    man_daily_ret = daily_ret(man_strat)
    bench_daily_ret = daily_ret(bench)
    learn_daily_ret = daily_ret(learn_strat)
    
    # Calculate standard deviation of daily returns
    man_std_dev = man_daily_ret.std()
    bench_std_dev = bench_daily_ret.std()
    learn_std_dev = learn_daily_ret.std()
    
    # Print results
    print("Manual Strategy Returns:")
    print(f"Cumulative Return: {man_cum_ret[0]}")
    # pdb.set_trace()
    print(f"Average Daily Return: " + str(cum_ret(man_strat.values).iloc[-1]/len(man_strat)))
    print(f"Standard Deviation of Daily Returns: {man_std_dev}")
    
    print("\nBenchmark Strategy Returns:")
    print(f"Cumulative Return: {bench_cum_ret[0]}")
    print(f"Average Daily Return: " + str(cum_ret(bench.values).iloc[-1]/len(bench)))
    print(f"Standard Deviation of Daily Returns: {bench_std_dev}")
    
    print("\nStrategy Learner Returns:")
    print(f"Cumulative Return: {learn_cum_ret[0]}")
    print(f"Average Daily Return: " + str(cum_ret(learn_strat.values).iloc[-1]/len(learn_strat)))
    print(f"Standard Deviation of Daily Returns: {learn_std_dev}")


def generate_plot(stock, manual, benchmark, strategy, in_sample=True):
    """
    Generate a plot comparing the performance of the manual strategy, benchmark, and strategy learner.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    manual = manual / manual.iloc[0]
    benchmark = benchmark / benchmark.iloc[0]
    strategy = strategy / strategy.iloc[0]
    ax.plot(manual, color='red', label='Manual Strategy')
    ax.plot(benchmark, color='purple', label='Benchmark')
    ax.plot(strategy, color='blue', label='Strategy Learner')
    ax.grid()
    sample_type = 'In-Sample' if in_sample else 'Out-of-Sample'
    ax.set_title(f'{stock} Strategy Comparison ({sample_type})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Portfolio Value')
    ax.legend()
    plt.savefig(f'{stock}_strategy_comparison_{sample_type.lower().replace("-", "_")}.png')
    plt.close(fig)
    

def run_experiment1(stock, sd, ed, sdout, edout, sv=100000, commission=9.95, impact=0.005):
    """
    Run Experiment 1: Compare the performance of a manual strategy with a strategy learner.
    """
    compare(stock, sd, ed, sv, commission, impact)


if __name__ == "__main__":
    # GT number
    np.random.seed(903969483)
    compare('JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000, commission=9.95, impact=0.005)