def KthQuantiles(S: list[int], k: int) -> list[int]:
    """
    Find the kth quantile, using D and C
    """
    n = len(S)
    
    # base case here, i mean if we only have 1, its the whole set
    if k == 1:
        return []
    
    # some helper functions
    def find_quants_helper(arr, positions):
        """
        use DnC with quick select
        """
        # blew this haha
        pos_s = positions
        # base case
        if not pos_s:
            return []
        
        # another base case, just grab the 1
        if len(pos_s) == 1:
            # quick select for single position
            return [quickselect(arr, pos_s[0])]
        
        # Find midd quantile pos
        middleidx = len(pos_s) // 2
        middlepos = pos_s[middleidx]
        
        # Partition array around the middle quantile
        # recursion here
        pivot_value = quickselect(arr[:], middlepos)
        
        # Partition into three parts: less, equal, greater
        # this is the 3 way split
        left = [xxx for xxx in arr if xxx < pivot_value]
        equal = [xxx for xxx in arr if xxx == pivot_value]
        right = [xxx for xxx in arr if xxx > pivot_value]
        
        # Recursive find quant in left and right
        leftpos = [ppp for ppp in pos_s if ppp < len(left)]
        rightpos = [ppp - len(left) - len(equal) for ppp in pos_s if ppp >= len(left) + len(equal)]
        
        reslt = []
        # use the helper recurse func
        reslt.extend(find_quants_helper(left, leftpos))
        
        # now find the mid to split on
        for ppp in pos_s:
            if len(left) <= ppp < len(left) + len(equal):
                reslt.append(pivot_value)
        reslt.extend(find_quants_helper(right, rightpos))
        return reslt
    
    # Calculate positions for k-1 quantiles (0-indexed)
    bucksize = n // k
    quantposs = [i * bucksize - 1 for i in range(1, k)]

    # Find quantiles at these positions
    reslt = find_quants_helper(S[:], quantposs)
    return sorted(reslt)


def quickselect(arr, k):
    """
    Find the k-th smallest el
    """
    # base case here
    if len(arr) == 1:
        # just return the 1
        return arr[0]
    
    # Choose pivot (using median-of-three for better performance)
    piv = medof3(arr)
    
    # Partition
    # same as above
    left = [xxx for xxx in arr if xxx < piv]
    equal = [xxx for xxx in arr if xxx == piv]
    right = [xxx for xxx in arr if xxx > piv]
    
    if k < len(left):
        # recursion here
        return quickselect(left, k)
    elif k < len(left) + len(equal):
        # just return the mid
        return equal[0]
    else:
        # recursion here again
        return quickselect(right, k - len(left) - len(equal))


def medof3(arr):
    """return median of first, middle, last"""
    # base case here
    if len(arr) <= 3:
        # just return the mid
        return sorted(arr)[len(arr) // 2]
    
    # just get the first, mid, last, simple stuff here to build up from
    first, mid, last = arr[0], arr[len(arr) // 2], arr[-1]
    return sorted([first, mid, last])[1]
