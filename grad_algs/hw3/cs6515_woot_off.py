def WootOff(
    W: tuple[int, ...], P: tuple[int, ...], C: tuple[int, ...]
) -> tuple[int, list[int]]:
    """
    So solving the woot off problem using dynamic programming.
    """
    n_prod = len(W)  # number of products
    cits = len(C)  # number of cities
    profits_per_city = []
    # this is the knapsack prob just applied to each of the citys
    for capacity_knap in C:
        # dp[w] = maximum profit achievable with weight capacity w
        dp_tab = [0] * (capacity_knap + 1)   # dp table
        # Build dp table
        for weight in range(1, capacity_knap + 1):
            # go thru each one
            for prod in range(n_prod):
                if W[prod] <= weight:
                    # ok, we can use this prod
                    dp_tab[weight] = max(dp_tab[weight], dp_tab[weight - W[prod]] + P[prod])   
        
        # max prof per city
        profits_per_city.append(dp_tab[capacity_knap])
    # grab all profs
    total_profit = sum(profits_per_city)
    return total_profit, profits_per_city
