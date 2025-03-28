""""""  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
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
  		  	   		 	 	 			  		 			     			  	 
import random as rand  		  	   		 	 	 			  		 			     			  	 
import numpy as np
import pdb		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
class QLearner(object):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    This is a Q learner object.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param num_states: The number of states to consider.  		  	   		 	 	 			  		 			     			  	 
    :type num_states: int  		  	   		 	 	 			  		 			     			  	 
    :param num_actions: The number of actions available..  		  	   		 	 	 			  		 			     			  	 
    :type num_actions: int  		  	   		 	 	 			  		 			     			  	 
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		 	 	 			  		 			     			  	 
    :type alpha: float  		  	   		 	 	 			  		 			     			  	 
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		 	 	 			  		 			     			  	 
    :type gamma: float  		  	   		 	 	 			  		 			     			  	 
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		 	 	 			  		 			     			  	 
    :type rar: float  		  	   		 	 	 			  		 			     			  	 
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		 	 	 			  		 			     			  	 
    :type radr: float  		  	   		 	 	 			  		 			     			  	 
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		 	 	 			  		 			     			  	 
    :type dyna: int  		  	   		 	 	 			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	 	 			  		 			     			  	 
    :type verbose: bool  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    def __init__(  		  	   		 	 	 			  		 			     			  	 
        self,  		  	   		 	 	 			  		 			     			  	 
        num_states=100,  		  	   		 	 	 			  		 			     			  	 
        num_actions=4,  		  	   		 	 	 			  		 			     			  	 
        alpha=0.2,  		  	   		 	 	 			  		 			     			  	 
        gamma=0.9,  		  	   		 	 	 			  		 			     			  	 
        rar=0.5,  		  	   		 	 	 			  		 			     			  	 
        radr=0.99,  		  	   		 	 	 			  		 			     			  	 
        dyna=0,  		  	   		 	 	 			  		 			     			  	 
        verbose=False,  		  	   		 	 	 			  		 			     			  	 
    ):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Constructor method  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        self.verbose = verbose  		  	   		 	 	 			  		 			     			  	 
        self.num_actions = num_actions
        self.num_states = num_states		  	   		 	 	 			  		 			     			  	 
        self.s = 0  		  	   		 	 	 			  		 			     			  	 
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.eps = rar
        self.eps_decay = radr
        self.dyna = dyna
        self.Q_table = np.zeros((self.num_states, self.num_actions))
        self.expr = {}      # experience for debugging
        # set for state, action, reward and new state
        self.expr['s'] = []
        self.expr['a'] = []
        self.expr['r'] = []
        self.expr['s_pr'] = []


    def author(self):
        """
        :return: The GT username of the student
        """
        return 'kmccarville3'


    def querysetstate(self, s):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Update the state without updating the Q-table  		  	   		 	 	 			  		 			     			  	 
                                                                                 
        :param s: The new state  		  	   		 	 	 			  		 			     			  	 
        :type s: int  		  	   		 	 	 			  		 			     			  	 
        :return: The selected action  		  	   		 	 	 			  		 			     			  	 
        :rtype: int  		  	   		 	 	 			  		 			     			  	 
        """
        if len(self.expr['s']) != len(self.expr['r']):
            self.expr['s'] = self.expr['s'][:-1]
            self.expr['a'] = self.expr['a'][:-1]
        self.s = s  		  	   		 	 	 			  		 			     			  	 
        action = rand.randint(0, self.num_actions - 1)		  	   		 	 	 			  		 			     			  	 
        if self.verbose:  		  	   		 	 	 			  		 			     			  	 
            print(f"s = {s}, a = {action}")
        self.expr['s'].append(s)    # set the state
        self.expr['a'].append(action)   # set the action	   		 	 	 			  		 			     			  	 
        return action  		  	   		 	 	 			  		 			     			  	 

                                                         
    def query(self, s_prime, r):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Update the Q table and return an action  		  	   		 	 	 			  		 			     			  	 
                                                                                 
        :param s_prime: The new state  		  	   		 	 	 			  		 			     			  	 
        :type s_prime: int  		  	   		 	 	 			  		 			     			  	 
        :param r: The immediate reward  		  	   		 	 	 			  		 			     			  	 
        :type r: float  		  	   		 	 	 			  		 			     			  	 
        :return: The selected action  		  	   		 	 	 			  		 			     			  	 
        :rtype: int  		  	   		 	 	 			  		 			     			  	 
        """
        self.expr['s_pr'].append(s_prime)   # set the new state
        self.expr['r'].append(r)   # set the reward
        # pdb.set_trace()

        # update Q table
        # self.Q_table[self.s, self.a] = (1 - self.alpha) * self.Q_table[self.s, self.a] + self.alpha * (r + self.gamma * np.max(self.Q_table[s_prime, :])) 		
        self.Q_table[self.expr['s'][-1], self.expr['a'][-1]] = (1 - self.alpha) * self.Q_table[self.expr['s'][-1], self.expr['a'][-1]] + self.alpha * (r + self.gamma * np.max(self.Q_table[s_prime, :]))

        # check if we are doing a random action
        if rand.random() < self.eps:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q_table[s_prime, :])
        
        # update the epsilon via decay rate
        self.eps = self.eps * self.eps_decay


        # Dyna-Q updates
        if self.dyna > 0:
            if len(self.expr['s']) != len(self.expr['r']):
                pdb.set_trace()
                print()
            self.run_dyna()
        
        # update the state and action
        self.expr['s'].append(s_prime)
        self.expr['a'].append(action)

        if self.verbose:  		  	   		 	 	 			  		 			     			  	 
            print(f"s = {s_prime}, a = {action}, r={r}") 

        return action  		  	   		 	 	 			  		 			     			  	 

    def run_dyna(self):
        """
        Perform Dyna-Q updates using simulated experience.
        """

        if len(self.expr['s']) < 200:
            return

        for _ in range(self.dyna):
            # Randomly sample from stored experience
            idx = rand.randint(0, len(self.expr['s']) - 1)
            s = self.expr['s'][idx]
            a = self.expr['a'][idx]
            try:
                r = self.expr['r'][idx]
            except Exception as e:
                pdb.set_trace()
                continue

            s_prime = self.expr['s_pr'][idx]

            # Update Q-table using the sampled experience
            self.Q_table[s, a] = (1 - self.alpha) * self.Q_table[s, a] + self.alpha * (r + self.gamma * np.max(self.Q_table[s_prime, :]))
