import numpy as np
from typing import Optional
from Random import Rand

def choose_from_bucket_with_probability(buckets: list[int]) -> int:
    """
    Choose an index from the buckets with probability proportional to the value of the bucket
    
    Args:
        buckets: a list of non-negative integers

    Returns: 
        the index of the chosen bucket
    """
    assert all(x >= 0 for x in buckets), "All the elements of the list should be non-negative"
    total = sum(buckets)
    choice: int = Rand.randint(0, total)
    for i, bucket in enumerate(buckets):
        if choice < bucket:
            return i
        choice -= bucket
    raise ValueError("Cannot happen")

def choose_from_bucket_with_prefix_probability(buckets: list[int], ind: Optional[int] = None) -> int:
    """
    Choose an index from the buckets with probability proportional to the value of the bucket
    
    Args:
        buckets: a list of monotonically increasing non-negative integers

    Returns: 
        the index of the chosen bucket

    Example:
        >>> choose_from_bucket_with_prefix_probability([3, 4, 6, 8])
        P[0] = 3/8, P[1] = 4/8-3/8, P[2] = 6/8-4/8, P[3] = 8/8-6/8
    """
    # not feasible, we want O(log n) time
    # assert all(x >= 0 for x in buckets), "All the elements of the list should be non-negative"
    
    total = buckets[-1]
    choice: int = ind if ind is not None else Rand.randint(0, total)

    # binary search
    left = 0
    right = len(buckets) - 1
    middle = lambda: (left + right) // 2

    while left < right:
        if choice < buckets[middle()]:
            right = middle()
        else:
            left = middle() + 1

    return left

## testing
# arr = [3, 4, 6, 8]
# for i in range(8):
#     print(choose_from_bucket_with_prefix_probability(arr, i))