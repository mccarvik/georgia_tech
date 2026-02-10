from decimal import Decimal
from typing import TypeVar

T = TypeVar("T", bool, Decimal, float, int, str)


def Select(input: list[T], k: int, start: int = 0, stop: int = None) -> int:
    """
    Returns the kth smallest value in the input list.

    This is implemented as QuickSelect rather than FastSelect due to the
    differences between conceptually optimal and practically optimal when
    considering execution performance.

    On average, the asymptotic performance is the same.
    """
    if stop is None:
        stop = len(input) - 1

    if start == stop:
        return input[start]

    pivot = Partition(input, start, stop)

    if k < pivot:
        return Select(input, k, start, pivot - 1)
    elif k > pivot:
        return Select(input, k, pivot + 1, stop)
    else:
        return input[k]


def Partition(input: list[T], start: int, stop: int) -> int:
    i = start

    mid = (start + stop) // 2

    pivot = input[mid]

    input[mid], input[stop] = input[stop], input[mid]

    for j in range(start, stop):
        if input[j] < pivot:
            if j != i:
                input[i], input[j] = input[j], input[i]

            i += 1

    input[i], input[stop] = input[stop], input[i]

    return i
