"""
Experiment 1: Compare the performance of a manual strategy with a strategy learner.
"""

import ManualStrategy as ms
import StrategyLearner as sl
import matplotlib.pyplot as plt
import marketsimcode
import datetime as dt
import pdb

def author():
    """
    Returns the author's name as a string.
    """
    return 'kmccarville3'


def compare(stock, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000, commission=9.95, impact=0.005):
    """
    Compare the performance of a manual strategy with a strategy learner.
    """
    manual, benchmark = ms.plot_manual_vs_benchmark("JPM", sd, ed, sv, True, commission, impact)
    learner = sl.StrategyLearner(impact=impact, commission=commission)
    learner.add_evidence(stock, sd, ed, sv)
    strat = learner.testPolicy(stock, sd, ed, sv)
    strat['Symbol']= 'JPM'
    strat.columns = ['Shares', 'Order', 'Symbol']
    strategy = marketsimcode.compute_portvals(strat, sv, commission, impact)
    generate_plot(stock, manual, benchmark, strategy)


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
    


if __name__ == "__main__":
    compare('JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000, commission=9.95, impact=0.005)