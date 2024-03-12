import random
from typing import List, TypeVar

T = TypeVar('T')

class Rand:
    # _instance = None
    # def __new__(cls, *args, **kwargs):
    #     if not cls._instance:
    #         cls._instance = super().__new__(cls, *args, **kwargs)
    #         cls
    
    @staticmethod
    def seed(a: int) -> None:
        random.seed(a)

    @staticmethod
    def randint(a: int, b: int) -> int:
        """
        Return a random integer N such that a <= N < b
        """
        return random.randint(a, b-1)
    
    @staticmethod
    def choice(seq: list[T]) -> T:
        return random.choice(seq)