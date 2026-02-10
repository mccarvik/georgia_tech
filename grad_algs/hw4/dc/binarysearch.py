from decimal import Decimal
from typing import TypeVar

T = TypeVar("T", bool, Decimal, float, int, str)


def BinarySearch(
    input: list[T], target: T, left: int = 0, right: int = None
) -> int | None:
    """
    Searches the input list for the presence of the given target.

    If found, the first matching index is returned. Otherwise, returns None.
    """
    if right is None:
        right = len(input) - 1

    while left <= right:
        mid = (left + right) // 2

        if input[mid] < target:
            left = mid + 1
        elif input[mid] > target:
            right = mid - 1
        else:
            return mid

    return None
