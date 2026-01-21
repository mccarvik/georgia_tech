"""
HW 0 - Kevin McCarville
"""

def long_inc_subs(seq):
    """
    so we are gonna have to do some recustion here to keep checking each subsequent number
    """
    # first base case
    if not seq:
        return 0
    
    # get the length of the seq
    n = len(seq)
    # so this is just a list for the longest from each index
    len_long_sub = [1] * n
    parent = [-1] * n
    
    # so we start at each index and then go thru the rest of the sequence continuosly looking for
    # increasing numbers
    for i_ind in range(1, n):
        for j_ind in range(i_ind):
            # If we can extend the subsequence ending at j
            if seq[j_ind] < seq[i_ind] and len_long_sub[j_ind] + 1 > len_long_sub[i_ind]:
                # keep track of the lengths
                len_long_sub[i_ind] = len_long_sub[j_ind] + 1
                # need to store the parent index so we can backtrack
                parent[i_ind] = j_ind
    

    max_l = max(len_long_sub)
    max_index = len_long_sub.index(max_l)
    # now we reconstruct the inds by backtracking through parent array
    ind_long_sub = []
    curr_ind = max_index
    while curr_ind != -1:
        ind_long_sub.append(curr_ind)
        curr_ind = parent[curr_ind]
    # reverse to get ascending order of indices
    ind_long_sub.reverse()
    return (max_l, ind_long_sub)


