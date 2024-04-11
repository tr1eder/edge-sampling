from heapdict import heapdict

class MinHeap:
    def __init__(self):
        self.pq = heapdict()

    def insert(self, key, priority):
        self.pq[key] = priority

    def decrease_key(self, key, new_priority):
        if key not in self.pq:
            raise KeyError("Key not found")
        if new_priority >= self.pq[key]:
            raise ValueError("New priority is not smaller")
        self.pq[key] = new_priority

    def extract_min(self):
        if not self.pq:
            raise IndexError("Priority queue is empty")
        key, priority = self.pq.popitem()
        return key, priority

    def is_empty(self):
        return len(self.pq) == 0
    
class MaxHeap:
    def __init__(self): 
        self.pq = heapdict()
    
    def insert(self, key, priority):
        self.pq[key] = -priority

    def increase_key(self, key, new_priority):
        if key not in self.pq:
            raise KeyError("Key not found")
        if new_priority <= -self.pq[key]:
            raise ValueError("New priority is not larger")
        self.pq[key] = -new_priority

    def extract_max(self):
        if not self.pq:
            raise IndexError("Priority queue is empty")
        key, priority = self.pq.popitem()
        return key
    
    def is_empty(self):
        return len(self.pq) == 0
    
    def toList(self):
        return list(self.pq)

    def __len__(self):
        return len(self.pq)
    
