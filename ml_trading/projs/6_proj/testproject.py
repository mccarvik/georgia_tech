"""
This is a test project file.
"""


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

    # get indicators
    