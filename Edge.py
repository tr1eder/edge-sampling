from abc import ABC, abstractmethod


class Edge(ABC):
    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass


class DEdge(Edge):
    def __init__(self, a: int, b: int) -> None:
        if a == b: raise ValueError("a and b cannot be equal")
        self.a = a
        self.b = b

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DEdge): return False
        return self.a == other.a and self.b == other.b
    
    def __hash__(self) -> int:
        return hash((self.a, self.b))
    
    def __repr__(self) -> str:
        return f"({self.a}, {self.b})"
    

class UEdge(Edge):
    # Rep Inv: self.a < self.b
    def __init__(self, a: int, b: int) -> None:
        if a == b: raise ValueError("a and b cannot be equal")
        self.a = min(a, b)
        self.b = max(a, b)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UEdge): return False
        return self.a == other.a and self.b == other.b
    
    def __hash__(self) -> int:
        return hash((self.a, self.b))

    def __repr__(self) -> str:
        return f"{{{self.a}, {self.b}}}"
    
## testing
# u1 = UEdge(1, 2)
# u2 = UEdge(2, 1)
# print (u1 == u2)