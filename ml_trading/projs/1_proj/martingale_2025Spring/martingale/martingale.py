""""""  		  	   		 	 	 			  		 			     			  	 
"""Assess a betting strategy.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
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
  		  	   		 	 	 			  		 			     			  	 
Student Name: Kevin McCarville 		  	   		 	 	 			  		 			     			  	 
GT User ID: kmccarville3 (replace with your User ID)  		  	   		 	 	 			  		 			     			  	 
GT ID: 903969483 (replace with your GT ID)  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 

import pdb  		  	   		 	 	 			  		 			     			  	 
import numpy as np
import matplotlib.pyplot as plt 		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
def author():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    :return: The GT username of the student  		  	   		 	 	 			  		 			     			  	 
    :rtype: str  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    return "kmccarville3"  # replace tb34 with your Georgia Tech username.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
def gtid():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    :return: The GT ID of the student  		  	   		 	 	 			  		 			     			  	 
    :rtype: int  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    return 903969483  # replace with your GT ID number  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
def get_spin_result(win_prob):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param win_prob: The probability of winning  		  	   		 	 	 			  		 			     			  	 
    :type win_prob: float  		  	   		 	 	 			  		 			     			  	 
    :return: The result of the spin.  		  	   		 	 	 			  		 			     			  	 
    :rtype: bool  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    result = False  		  	   		 	 	 			  		 			     			  	 
    if np.random.random() <= win_prob:  		  	   		 	 	 			  		 			     			  	 
        result = True  		  	   		 	 	 			  		 			     			  	 
    return result  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
def first_experiment(epis, spins, win_prob, bet_amt=1):
    """
    Run the first experiment
    """
    # setup the results array
    # add 0 to start chart at 0
    results = [epis, spins + 1]
    # set np zeros
    results = np.zeros(results)

    # Outer loop for episodes
    for i in range(epis):
        win = False
        bet = bet_amt
        # inner loop for spins
        # check if value is over 80 or past spin count
        ctr = 0
        while ctr < spins:
            if results[i][ctr] >= 80:
                # We won
                ctr += 1
                results[i][ctr] = results[i][ctr - 1]
            else:
                ctr += 1
                spin_res = get_spin_result(win_prob)
                if spin_res:
                    results[i][ctr] = results[i][ctr - 1] + bet
                    bet = bet_amt
                else:
                    results[i][ctr] = results[i][ctr - 1] - bet
                    bet = bet * 2
            # print(bet)
    print(results)
    return results


def second_experiment(epis, spins, win_prob, bet_amt=1):
    """
    Run the second experiment
    """
    # setup the results array
    # add 0 to start chart at 0
    results = [epis, spins + 1]
    # set np zeros
    results = np.zeros(results)

    # Outer loop for episodes
    for i in range(epis):
        bet = bet_amt
        # inner loop for spins
        # check if value is over 80 or past spin count
        ctr = 0
        while ctr < spins:
            bet = check_bet(bet, results[i][ctr])
            if results[i][ctr] >= 80 or results[i][ctr] <= -256:
                # We won
                ctr += 1
                results[i][ctr] = results[i][ctr - 1]
            else:
                ctr += 1
                spin_res = get_spin_result(win_prob)
                if spin_res:
                    results[i][ctr] = results[i][ctr - 1] + bet
                    bet = bet_amt
                else:
                    results[i][ctr] = results[i][ctr - 1] - bet
                    bet = bet * 2
            # print(bet)
    print(results)
    return results


def check_bet(bet, result):
    """
    Check the bet for the second experiment
    """
    bet = min(bet, 256-result)
    return bet


def fig1(results):
    """
    Create the first figure
    """
    plt.figure()
    for i in range(results.shape[0]):
        plt.plot(results[i])
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.xlabel('Spins')
    plt.ylabel('Value')
    plt.title('Martingale Strategy: All Episodes')
    plt.savefig('images/fig1.png')
    # plt.show()


def fig2(results, median=False, exp2=False):
    """
    Create the second figure
    """
    # diff_results = np.diff(results, axis=1)
    # pdb.set_trace()
    if not median:
        mean_values = np.mean(results, axis=0)
    else:
        mean_values = np.median(results, axis=0)
    std_dev = np.std(results, axis=0)
    plt.figure()
    if not median:
        plt.plot(mean_values, label='Mean')
        plt.plot(mean_values + std_dev, label='Mean + Std Dev', linestyle='--')
        plt.plot(mean_values - std_dev, label='Mean - Std Dev', linestyle='--')
        plt.title('Martingale Strategy: Mean and Std Dev')
    else:
        plt.plot(mean_values, label='Median')
        plt.plot(mean_values + std_dev, label='Median + Std Dev', linestyle='--')
        plt.plot(mean_values - std_dev, label='Median - Std Dev', linestyle='--')
        plt.title('Martingale Strategy: Median and Std Dev')
    
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.xlabel('Spins')
    plt.ylabel('Value')
    plt.legend()
    if not exp2:
        if not median:
            plt.savefig('images/fig2.png')
        else:
            plt.savefig('images/fig2_median.png')
    else:
        if not median:
            plt.savefig('images/fig3.png')
        else:
            plt.savefig('images/fig3_median.png')
    # plt.show()


def test_code():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    Method to test your code  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    win_prob = 18/38  # 16 black numbers out of 38 numbers because 0 and 00 are green (American wheel)
    # win_prob = 0.6
    np.random.seed(gtid())  # do this only once  		  	   		 	 	 			  		 			     			  	 
    print(get_spin_result(win_prob))  # test the roulette spin  		  	   		 	 	 			  		 			     			  	 
    # add your code here to implement the experiments  	
    results = first_experiment(10, 1000, win_prob)
    res2 = first_experiment(1000, 1000, win_prob)
    fig1(results)
    fig2(res2)
    fig2(res2, median=True)

    res3 = second_experiment(1000, 1000, win_prob)
    fig2(res3, exp2=True)
    fig2(res3, median=True, exp2=True)

  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    test_code()