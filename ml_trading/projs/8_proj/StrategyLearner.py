""""""  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
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
import random  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import pandas as pd 
import numpy as np 		  	   		 	 	 			  		 			     			  	 
import util 

import BagLearner as bl
import RTLearner as rt
import indicators as ind

from ManualStrategy import ema_signal, reconstruct_bollinger_bands
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
class StrategyLearner(object):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	 	 			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		 	 	 			  		 			     			  	 
    :type verbose: bool  		  	   		 	 	 			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 	 	 			  		 			     			  	 
    :type impact: float  		  	   		 	 	 			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 	 	 			  		 			     			  	 
    :type commission: float  		  	   		 	 	 			  		 			     			  	 
    """

    def author(self):
        """
        Returns the author of this code
        """
        return "kmccarville3"


    # constructor  		  	   		 	 	 			  		 			     			  	 
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Constructor method  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        self.verbose = verbose  		  	   		 	 	 			  		 			     			  	 
        self.impact = impact  		  	   		 	 	 			  		 			     			  	 
        self.commission = commission  
        self.leaf_size = 5
        self.look_back = 21
        self.bags = 20
        self.learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": self.leaf_size}, bags=self.bags, boost = False, verbose = False)


    # this method should create a QLearner, and train it for trading  		  	   		 	 	 			  		 			     			  	 
    def add_evidence(  		  	   		 	 	 			  		 			     			  	 
        self,  		  	   		 	 	 			  		 			     			  	 
        symbol="IBM",  		  	   		 	 	 			  		 			     			  	 
        sd=dt.datetime(2008, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        ed=dt.datetime(2009, 12, 31),  		  	   		 	 	 			  		 			     			  	 
        sv=100000,  		  	   		 	 	 			  		 			     			  	 
    ):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Trains your strategy learner over a given time frame.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        :param symbol: The stock symbol to train on  		  	   		 	 	 			  		 			     			  	 
        :type symbol: str  		  	   		 	 	 			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 			  		 			     			  	 
        :type sd: datetime  		  	   		 	 	 			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 			  		 			     			  	 
        :type ed: datetime  		  	   		 	 	 			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	 	 			  		 			     			  	 
        :type sv: int  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        # add your code to do learning here
        prices = util.get_data([symbol], pd.date_range(sd, ed))  # automatically adds SPY
        pxs = prices[symbol]

        x_vals = self.generate_feats(pxs)
        data_x = x_vals[:-self.look_back].values

        min_return = 0.01
        data_y = np.zeros((data_x.shape[0], 1))
        returns = pxs.values[self.look_back:] / pxs.values[:-self.look_back] - 1
        for i in range(0, data_x.shape[0]):
            if returns[i] > min_return + self.impact:
                data_y[i] = 1
            elif returns[i] < -min_return - self.impact:
                data_y[i] = -1
            else:
                data_y[i] = 0
        self.learner.add_evidence(data_x, data_y)


    def testPolicy(  		  	   		 	 	 			  		 			     			  	 
        self,  		  	   		 	 	 			  		 			     			  	 
        symbol="IBM",  		  	   		 	 	 			  		 			     			  	 
        sd=dt.datetime(2009, 1, 1),  		  	   		 	 	 			  		 			     			  	 
        ed=dt.datetime(2010, 12, 31),  		  	   		 	 	 			  		 			     			  	 
        sv=10000,  		  	   		 	 	 			  		 			     			  	 
    ):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		 	 	 			  		 			     			  	 
        :type symbol: str  		  	   		 	 	 			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 			  		 			     			  	 
        :type sd: datetime  		  	   		 	 	 			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 			  		 			     			  	 
        :type ed: datetime  		  	   		 	 	 			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	 	 			  		 			     			  	 
        :type sv: int  		  	   		 	 	 			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		 	 	 			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		 	 	 			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		 	 	 			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		 	 	 			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        dates = pd.date_range(sd, ed)
        prices = util.get_data([symbol], dates)
        pxs = prices[symbol]

        test_x = self.generate_feats(pxs)
        test_x = test_x.dropna()
        test_y = self.learner.query(test_x.values)
        trades = prices.copy() * 0

        holdings = 0
        trades.columns = ['Shares', 'Order']

        for i in range(len(test_y)):
            # Default to HOLD
            shares = 0
            order = 'HOLD'
            
            # buy
            if test_y[i] > 0.10:
                if holdings == 0:
                    shares = 1000
                    order = 'BUY'
                    holdings = 1
                elif holdings == -1:
                    shares = 2000
                    order = 'BUY'
                    holdings = 1
            # sell
            elif test_y[i] < -0.1:
                if holdings == 0:
                    shares = -1000
                    order = 'SELL'
                    holdings = -1
                elif holdings == 1:
                    shares = -2000
                    order = 'SELL'
                    holdings = -1
            # neutral
            else:
                if holdings == 1:
                    shares = -1000
                    order = 'SELL'
                    holdings = 0
                elif holdings == -1:
                    shares = 1000
                    order = 'BUY'
                    holdings = 0
            
            # Update the DataFrame correctly using loc
            trades.loc[trades.index[i], 'Shares'] = shares
            trades.loc[trades.index[i], 'Order'] = order
            
            print(f"Day {i}: Holdings={holdings}, Shares={shares}, Order={order}, Test_y={test_y[i]}")

        # pdb.set_trace()
        # Drop all rows where orders equal 0
        trades = trades[trades['Order'] != 0]
        return trades


    def generate_feats(self, price_df):
        """
        Generates features for the model
        :param price_df: DataFrame of prices
        :return: DataFrame of features
        """
        rsi = ind.calculate_rsi(price_df, self.look_back)
        bboll = ind.calculate_bollinger_bands(price_df, self.look_back)
        # bboll = reconstruct_bollinger_bands(bboll, price_df, self.look_back)
        ema = ind.calculate_ema(price_df, self.look_back)
        # ema = ema_signal(ema, price_df, "JPM")
        # create a new dataframe to hold the features
        features_df = pd.DataFrame({
            'rsi': rsi,
            'ema': ema,
            'bboll': bboll
        })
        return features_df




if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    print("One does not simply think up a strategy")  		  	   		 	 	 			  		 			     			  	 
