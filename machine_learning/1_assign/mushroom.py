"""
Assignment 1 - main run method for mushroom dataset
"""

import pandas as pd

DATA_PATH = "data/mushroom/mushrooms.csv"

def run():
    """
    main run method for the script
    """
    df = pd.read_csv(DATA_PATH)
    print(df.head())



if __name__ == "__main__":
    run()