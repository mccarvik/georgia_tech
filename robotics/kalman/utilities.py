from math import sqrt

def clamp(value, lower, upper):
    """Clamp value to be in the range [lower, upper].

    This function is from this StackOverflow answer:
    https://stackoverflow.com/questions/9775731/clamping-floating-numbers-in-python/58470178#58470178
    """
    # Beginning of code from
    # https://stackoverflow.com/questions/9775731/clamping-floating-numbers-in-python/58470178#58470178
    return lower if value < lower else upper if value > upper else value
    # End of code from
    # https://stackoverflow.com/questions/9775731/clamping-floating-numbers-in-python/58470178#58470178

def distance_formula(pt_0, pt_1):
    """Compute the 2D distance between point 0 and point 1."""
    return sqrt((pt_1[0] - pt_0[0])**2 + (pt_1[1] - pt_0[1])**2)
