def HW0(A: tuple[int, ...]) -> tuple[int, list[int]]:
    """
    so we are gonna have to do some recustion here to keep checking each subsequent number
    """

    
    # did it in the wrong file at first
    seq = A

    # start with base case
    if not seq:
        return (0, [])
    
    # Base case: single element
    if len(seq) == 1:
        return (1, [0])
    
    # track the longest lenntgh so far
    max_len = 1
    max_st_ind = 0
    
    # Track the current  seq
    curr_len = 1
    curr_st_ind = 0
    
    for ind in range(1, len(seq)):
        # simple just check if its longer than the last
        if seq[ind] > seq[ind - 1]:
            #  we add to the lnegth
            curr_len += 1
        else:
            # streak is broken, is it hte longest?
            if curr_len > max_len:
                max_len = curr_len
                max_st_ind = curr_st_ind
            # Start a new sequence from current index
            # we actually dont have to backtrack as we know any sequence starting from
            # and intermediate has to be shorter than the longest so far
            curr_len = 1
            curr_st_ind = ind
    
    # Check one final time after loop ends
    # same as we did before
    # terrible typo here
    if curr_len > max_len:
        max_len = curr_len
        max_st_ind = curr_st_ind
    
    # this just gradas the inds
    indices = list(range(max_st_ind, max_st_ind + max_len))
    return (max_len, indices)
