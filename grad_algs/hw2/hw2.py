# HW2
# Kevin McCarville

def max_profit_divconq(rev, cost):
    # grab the length of the arr
    nnn = len(rev)
    # just set this to 0 for now
    pref_arr = [0] * (nnn+1)
    # get our one day answer for each of these
    for iii in range(1, nnn+1):
        pref_arr[iii] = rev[iii-1] - cost[iii-1] + pref_arr[iii-1]

    def divconq(left, right):
        # so this is our base case where we use the same day so profit is 0
        if left == right:
            # profit = revenue[left-1] - revenue[left-2] - cost[left-1]  (but we use prefix)
            return 0, left, left, pref_arr[left-1], pref_arr[right]

        # Grab the middle, I mean we are dividing and conquering
        mid = (left + right) // 2
        # and now we go both directions
        # Left guy
        left_prof, left_i, left_j, left_min_p, left_max_p = divconq(left, mid)
        # Right guy
        right_prof, right_i, right_j, right_min_p, right_max_p = divconq(mid+1, right)

        # grab the better way
        best_profit = max(left_prof, right_prof)
        # best left guy
        best_left = left_i if left_prof >= right_prof else right_i
        # best right guy
        best_right = left_j if left_prof >= right_prof else right_j

        # Cross - we only need the left guy's min and the right guy's max cuz its a profit
        cross_min_prof = min(left_min_p, right_min_p)
        cross_max_prof = max(left_max_p, right_max_p)
        # so the prof becomes
        cross_profit = right_max_p - left_min_p

        # and if this our best yet, save it
        if cross_profit > best_profit:
            best_profit = cross_profit
        return best_profit, best_left, best_right, cross_min_prof, cross_max_prof

    # actually call the func from the start
    # dont need to keep track of the min and max at this point, just the days and prof
    max_prof, i, j, _, _ = divconq(1, nnn)
    return max_prof, i, j