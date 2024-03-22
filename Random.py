import random
from typing import List, TypeVar, Callable

T = TypeVar('T')

class Rand:
    rng: random.Random

    @staticmethod
    def seed(seed: int) -> None:
        Rand.rng = random.Random(seed)

    @staticmethod
    def randint(a: int, b: int) -> int:
        """
        Return a random integer N such that a <= N < b
        """
        return Rand.rng.randint(a, b-1)
    
    @staticmethod
    def choice(seq: list[T]) -> T:
        return Rand.rng.choice(seq)
    
    @staticmethod
    def random() -> float:
        return Rand.rng.random()
    
    @staticmethod
    def isDeterministic(function: Callable[[], str], tries: int) -> bool:
        """ Check if a function is deterministic with tries many trials"""
        results = [function() for _ in range(tries)]
        return all(x == results[0] for x in results)
    
if __name__ == "__main__":
    Rand.seed(42)  # Set the seed
    print(Rand.randint(0, 10))  # Should always print the same number
    print(Rand.randint(0, 10))  # Should always print the same number
