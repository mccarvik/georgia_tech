import ManualStrategy as ms
import StrategyLearner as sl
import experiment1 as e1
import experiment2 as e2

import datetime as dt
import numpy as np
import pandas as pd

def author():
    """
    Returns the author's name as a string.
    """
    return 'kmccarville3'

np.random.seed(903969483)

stock = "JPM"

sd_insample = dt.datetime(2008, 1, 1)
ed_insample = dt.datetime(2009, 12, 31)
sv_insample = 100000
commission = 9.95
impact = 0.005
sd_outofsample = dt.datetime(2010, 1, 1)
ed_outofsample = dt.datetime(2011, 12, 31)

ms.run_manual_strat(stock, sd_insample, ed_insample, sd_outofsample, ed_outofsample, sv_insample, commission, impact)

e1.run_experiment1(stock, sd_insample, ed_insample, sd_outofsample, ed_outofsample, sv_insample, commission, impact)

e2.run_experiment2(stock, sd_insample, ed_insample, sd_outofsample, ed_outofsample, sv_insample, commission, impact)
