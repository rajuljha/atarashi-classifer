import numpy as np
from collections import defaultdict

class SimHash:
    def __init__(self, hash_size, input_dim):
        self.hash_size = hash_size
        self.random_projection = np.random.normal(loc=0, scale=2.0, size=(hash_size, input_dim))

    def _hash(self, vector):
        projected_vector = np.dot(self.random_projection, vector)
        hash_bits = np.zeros(self.hash_size)
        for i in range(self.hash_size):
            if projected_vector[i] > 1:
                hash_bits[i] += 1
            else:
                hash_bits[i] -= 1
        return ''.join(['1' if bit > 0 else '0' for bit in hash_bits])

    def compute(self, vector):
        return self._hash(vector)


class LSH:
    def __init__(self, hash_size, input_dim, num_tables):
        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_tables = num_tables
        self.tables = [defaultdict(list) for _ in range(num_tables)]
        self.hash_functions = [SimHash(hash_size, input_dim) for _ in range(num_tables)]

    def _hash(self, vector, table_idx):
        return self.hash_functions[table_idx].compute(vector)

    def add(self, vector, word):
        for i in range(self.num_tables):
            hash_value = self._hash(vector, i)
            self.tables[i][hash_value].append(word)
            if len(self.tables[i][hash_value]) > 1 and len(self.tables[i][hash_value]) <= 5:
                print(f"[DEBUG] Collision in table {i}, bucket {hash_value[:8]}... → {len(self.tables[i][hash_value])} items")

    def query(self, vector):
        candidates = set()
        for i in range(self.num_tables):
            hash_value = self._hash(vector, i)
            matches = self.tables[i].get(hash_value, [])
            if matches:
                print(f"[DEBUG] Table {i} hit → {len(matches)} candidates")
            candidates.update(matches)
        return candidates
