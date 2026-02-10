from decimal import Decimal
from typing import TypeVar

T = TypeVar("T", bool, Decimal, float, int, str)


def MergeSort(input: list[T], start: int = 0, stop: int = None) -> list[T]:
    """
    Sorts the values in the input list between start and stop in ascending order.

    Returns a new list containing only those sorted values.
    """
    if stop is None:
        stop = len(input) - 1

    if start == stop:
        return [input[start]]

    mid = (start + stop) // 2

    return Merge(
        MergeSort(input, start, mid),
        MergeSort(input, mid + 1, stop),
    )


def Merge(left: list[T], right: list[T]) -> list[T]:
    result = []

    i = 0
    j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    while i < len(left):
        result.append(left[i])
        i += 1

    while j < len(right):
        result.append(right[j])
        j += 1

    return result
