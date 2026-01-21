def HW0(A: tuple[int, ...]) -> tuple[int, list[int]]:
    n = len(A)

    if n == 0:
        return 0, []

    t = []
    parent = []

    for i in range(n):
        t.append(1)
        parent.append(-1)

        for j in range(i):
            if (A[j] < A[i]) and (t[i] < 1 + t[j]):
                t[i] = 1 + t[j]
                parent[i] = j

    length = max(t)
    max_index = t.index(length)

    positions = []
    curr_ind = max_index
    
    while curr_ind != -1:
        positions.append(curr_ind)
        curr_ind = parent[curr_ind]

    positions.reverse()

    return length, positions
