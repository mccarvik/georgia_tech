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


def author():
    """
    :return: The GT username of the student
    """
    return 'kmccarville3'


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
    pdb.set_trace()
    port_val = port_val / port_val.iloc[0]
    jpm = 1000 * util.get_data(["JPM"], pd.date_range(sd, ed), addSPY=False, colname="Adj Close").dropna()
    pdb.set_trace()
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
    plt.savefig('Theoretically Optimal Strategy vs Benchmark.png')
    plt.close()


