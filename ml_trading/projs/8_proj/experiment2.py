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



def compare_comms(stock, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000, commission=9.95, impact=0.005):
    """
    Compare the performance of a manual strategy with a strategy learner.
    """
    # keeping commission the same
    commission = -100
    manual, benchmark = ms.plot_manual_vs_benchmark(stock, sd, ed, sv, True, commission, impact)
    learner = sl.StrategyLearner(impact=impact, commission=commission)
    learner.add_evidence(stock, sd, ed, sv)
    strat = learner.testPolicy(stock, sd, ed, sv)
    strat['Symbol']= stock
    strat['Order'] = strat['Shares'].apply(lambda x: 'BUY' if x > 0 else ('SELL' if x < 0 else 'HOLD'))
    strat.columns = ['Shares', 'Symbol', 'Order']
    strategy = marketsimcode.compute_portvals(strat, sv, commission, impact)
    print_strategy_returns(manual, benchmark, strategy)

    
    # keeping impact the same
    # set commission to 500
    commission = 9.95
    manual2, benchmark2 = ms.plot_manual_vs_benchmark(stock, sd, ed, sv, True, commission, impact)
    learner2 = sl.StrategyLearner(impact=impact, commission=commission)
    learner2.add_evidence(stock, sd, ed, sv)
    strat2 = learner2.testPolicy(stock, sd, ed, sv)
    strat2['Symbol']= stock
    strat2['Order'] = strat2['Shares'].apply(lambda x: 'BUY' if x > 0 else ('SELL' if x < 0 else 'HOLD'))
    strat2.columns = ['Shares', 'Symbol', 'Order']
    strategy2 = marketsimcode.compute_portvals(strat2, sv, commission, impact)
    print_strategy_returns(manual2, benchmark2, strategy2)

    # keeping impact the same
    # set impact to 1%
    commission = 500
    manual3, benchmark3 = ms.plot_manual_vs_benchmark(stock, sd, ed, sv, True, commission, impact)
    learner3 = sl.StrategyLearner(impact=impact, commission=commission)
    learner3.add_evidence(stock, sd, ed, sv)
    strat3 = learner2.testPolicy(stock, sd, ed, sv)
    strat3['Symbol']= stock
    strat3['Order'] = strat3['Shares'].apply(lambda x: 'BUY' if x > 0 else ('SELL' if x < 0 else 'HOLD'))
    strat3.columns = ['Shares', 'Symbol', 'Order']
    strategy3 = marketsimcode.compute_portvals(strat3, sv, commission, impact)
    print_strategy_returns(manual3, benchmark3, strategy3)

    generate_plot_comms(
        stock,
        manual,
        manual2,
        manual3,
        benchmark,
        benchmark2,
        benchmark3,
        strategy,
        strategy2,
        strategy3,
        in_sample=False
    )


def generate_plot_comms(stock, manual1, manual2, manual3, benchmark1, benchmark2, benchmark3, strategy1, strategy2, strategy3, in_sample=True):
    """
    Generate a plot to compare the performance of manual, benchmark, and strategy learner portfolios.

    Parameters:
    - stock: The stock symbol
    - manual1, manual2, manual3: DataFrames of manual strategy portfolio values for different commissions
    - benchmark1, benchmark2, benchmark3: DataFrames of benchmark strategy portfolio values for different commissions
    - strategy1, strategy2, strategy3: DataFrames of strategy learner portfolio values for different commissions
    - in_sample: Boolean indicating if the data is in-sample or out-of-sample
    """
    plt.figure(figsize=(10, 6))

    # Plot manual strategy
    plt.plot(manual1.index, manual1 / manual1.iloc[0], label="Manual Strategy (Commission=-100)", color="blue", linestyle="-")
    plt.plot(manual2.index, manual2 / manual2.iloc[0], label="Manual Strategy (Commission=9.95)", color="blue", linestyle="--")
    plt.plot(manual3.index, manual3 / manual3.iloc[0], label="Manual Strategy (Commission=500)", color="blue", linestyle=":")

    # Plot benchmark strategy
    plt.plot(benchmark1.index, benchmark1 / benchmark1.iloc[0], label="Benchmark Strategy (Commission=-100)", color="green", linestyle="-")
    plt.plot(benchmark2.index, benchmark2 / benchmark2.iloc[0], label="Benchmark Strategy (Commission=9.95)", color="green", linestyle="--")
    plt.plot(benchmark3.index, benchmark3 / benchmark3.iloc[0], label="Benchmark Strategy (Commission=500)", color="green", linestyle=":")

    # Plot strategy learner
    plt.plot(strategy1.index, strategy1 / strategy1.iloc[0], label="Strategy Learner (Commission=-100)", color="red", linestyle="-")
    plt.plot(strategy2.index, strategy2 / strategy2.iloc[0], label="Strategy Learner (Commission=9.95)", color="red", linestyle="--")
    plt.plot(strategy3.index, strategy3 / strategy3.iloc[0], label="Strategy Learner (Commission=500)", color="red", linestyle=":")

    # Add labels, title, and legend
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.title(f"Comparison of Strategies for {stock} ({'In-Sample' if in_sample else 'Out-of-Sample'})")
    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"comparison_{stock}_{'in_sample' if in_sample else 'out_of_sample'}_commission.png")

    # Show the plot
    # plt.show()


def compare(stock, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000, commission=9.95, impact=0.005):
    """
    Compare the performance of a manual strategy with a strategy learner.
    """
    # keeping impact the same
    manual, benchmark = ms.plot_manual_vs_benchmark(stock, sd, ed, sv, True, commission, impact)
    learner = sl.StrategyLearner(impact=impact, commission=commission)
    learner.add_evidence(stock, sd, ed, sv)
    strat = learner.testPolicy(stock, sd, ed, sv)
    strat['Symbol']= stock
    strat['Order'] = strat['Shares'].apply(lambda x: 'BUY' if x > 0 else ('SELL' if x < 0 else 'HOLD'))
    strat.columns = ['Shares', 'Symbol', 'Order']
    strategy = marketsimcode.compute_portvals(strat, sv, commission, impact)
    print_strategy_returns(manual, benchmark, strategy)

    
    # set impact to 0
    impact = 0
    manual2, benchmark2 = ms.plot_manual_vs_benchmark(stock, sd, ed, sv, True, commission, impact)
    learner2 = sl.StrategyLearner(impact=impact, commission=commission)
    learner2.add_evidence(stock, sd, ed, sv)
    strat2 = learner2.testPolicy(stock, sd, ed, sv)
    strat2['Symbol']= stock
    strat2['Order'] = strat2['Shares'].apply(lambda x: 'BUY' if x > 0 else ('SELL' if x < 0 else 'HOLD'))
    strat2.columns = ['Shares', 'Symbol', 'Order']
    strategy2 = marketsimcode.compute_portvals(strat2, sv, commission, impact)
    print_strategy_returns(manual2, benchmark2, strategy2)

    # set impact to 1%
    impact = 0.02
    manual3, benchmark3 = ms.plot_manual_vs_benchmark(stock, sd, ed, sv, True, commission, impact)
    learner3 = sl.StrategyLearner(impact=impact, commission=commission)
    learner3.add_evidence(stock, sd, ed, sv)
    strat3 = learner2.testPolicy(stock, sd, ed, sv)
    strat3['Symbol']= stock
    strat3['Order'] = strat3['Shares'].apply(lambda x: 'BUY' if x > 0 else ('SELL' if x < 0 else 'HOLD'))
    strat3.columns = ['Shares', 'Symbol', 'Order']
    strategy3 = marketsimcode.compute_portvals(strat3, sv, commission, impact)
    print_strategy_returns(manual3, benchmark3, strategy3)

    generate_plot(
        stock,
        manual,
        manual2,
        manual3,
        benchmark,
        benchmark2,
        benchmark3,
        strategy,
        strategy2,
        strategy3,
        in_sample=False
    )


def generate_plot(stock, manual1, manual2, manual3, benchmark1, benchmark2, benchmark3, strategy1, strategy2, strategy3, in_sample=True):
    """
    Generate a plot to compare the performance of manual, benchmark, and strategy learner portfolios.

    Parameters:
    - stock: The stock symbol
    - manual1, manual2, manual3: DataFrames of manual strategy portfolio values for different impacts
    - benchmark1, benchmark2, benchmark3: DataFrames of benchmark strategy portfolio values for different impacts
    - strategy1, strategy2, strategy3: DataFrames of strategy learner portfolio values for different impacts
    - in_sample: Boolean indicating if the data is in-sample or out-of-sample
    """
    plt.figure(figsize=(10, 6))

    # Plot manual strategy
    plt.plot(manual1.index, manual1 / manual1.iloc[0], label="Manual Strategy (Impact=0.005)", color="blue", linestyle="-")
    plt.plot(manual2.index, manual2 / manual2.iloc[0], label="Manual Strategy (Impact=0)", color="blue", linestyle="--")
    plt.plot(manual3.index, manual3 / manual3.iloc[0], label="Manual Strategy (Impact=0.02)", color="blue", linestyle=":")

    # Plot benchmark strategy
    plt.plot(benchmark1.index, benchmark1 / benchmark1.iloc[0], label="Benchmark Strategy (Impact=0.005)", color="green", linestyle="-")
    plt.plot(benchmark2.index, benchmark2 / benchmark2.iloc[0], label="Benchmark Strategy (Impact=0)", color="green", linestyle="--")
    plt.plot(benchmark3.index, benchmark3 / benchmark3.iloc[0], label="Benchmark Strategy (Impact=0.02)", color="green", linestyle=":")

    # Plot strategy learner
    plt.plot(strategy1.index, strategy1 / strategy1.iloc[0], label="Strategy Learner (Impact=0.005)", color="red", linestyle="-")
    plt.plot(strategy2.index, strategy2 / strategy2.iloc[0], label="Strategy Learner (Impact=0)", color="red", linestyle="--")
    plt.plot(strategy3.index, strategy3 / strategy3.iloc[0], label="Strategy Learner (Impact=0.02)", color="red", linestyle=":")

    # Add labels, title, and legend
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.title(f"Comparison of Strategies for {stock} ({'In-Sample' if in_sample else 'Out-of-Sample'})")
    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"comparison_{stock}_{'in_sample' if in_sample else 'out_of_sample'}_impact.png")

    # Show the plot
    # plt.show()


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
    

def run_experiment2(stock, sd_insample, ed_insample, sd_outofsample, ed_outofsample, sv_insample, commission, impact):
    """
    Run the experiment for the given stock and parameters.
    """
    # In-sample analysis
    compare(stock, sd=sd_insample, ed=ed_insample, sv=sv_insample, commission=commission, impact=impact)

    compare_comms(stock, sd=sd_insample, ed=ed_insample, sv=sv_insample, commission=commission, impact=impact)


if __name__ == "__main__":
    # GT number
    np.random.seed(903969483)
    compare('JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000, commission=9.95, impact=0.005)
    compare_comms('JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000, commission=9.95, impact=0.005)