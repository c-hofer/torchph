from collections import Counter, defaultdict
import numpy as np
from simplicial_complex import boundary_operator

class SortedListBoundaryMatrix:
    
    def __init__(self, filtrated_sc):
        self._data = None
        self._simplex_dims = None    
        self.stats = defaultdict(list)
        self._init_from_filtrated_sp(filtrated_sc)
        self.max_pairs = 100

    def _init_from_filtrated_sp(self, filtrated_sp):
        simplex_to_ordering_position = {s: i for i, s in enumerate(filtrated_sp)}
        
        self._data = [[]]*len(filtrated_sp)
        self._simplex_dims = {}

        for col_i, s in enumerate(filtrated_sp):
            boundary = boundary_operator(s)
            orderings_of_boundaries = sorted((simplex_to_ordering_position[b] for b in boundary))
            self._data[col_i] = orderings_of_boundaries
            self._simplex_dims[col_i] = len(s) - 1 
                
        
    
    def add_column_i_to_j(self, i, j):  
        col_i = self._data[i]
        col_j = self._data[j]
               
        self._data[j] = sorted((k for k, v in Counter(col_i + col_j).items() if v == 1))
        
    def get_pivot(self):
        return {col_i: col_i_entries[-1] for col_i, col_i_entries in enumerate(self._data)
                if len(col_i_entries) > 0}
    
    def pairs_for_reduction(self):
        tmp = defaultdict(list)
        for column_i, pivot_i in self.get_pivot().items():
            tmp[pivot_i].append(column_i)
            
        ret = []
        for k, v in tmp.items():
            if len(v) > 1:
                for j in v[1:]:
                    ret.append((v[0], j))
            
        if self.max_pairs is not None:
            ret = ret[:self.max_pairs]
        return ret
    
    def reduction_step(self):
        pairs = self.pairs_for_reduction()
        
        if len(pairs) == 0:
            raise StopIteration()            
        
        self.stats['n_pairs'].append(len(pairs))
        
        for i, j in pairs:
            self.add_column_i_to_j(i, j)              
                
        self.stats['longest_column'].append(max(len(c) for c in self._data))
        
    def reduce(self):
        iterations = 0
        try:
            while True:
                self.reduction_step()
                iterations += 1
                
        except StopIteration:
            print('Reached end of reduction after ', iterations, ' iterations')
        
    def row_i_contains_lowest_one(self):
        pass
    
    def read_barcodes(self):
        assert len(self.pairs_for_reduction()) == 0, 'Matrix is not reduced'
        
        barcodes = defaultdict(list)
        pivot = self.get_pivot()
        
        for death, birth in pivot.items():
            assert birth < death
            barcodes[self._simplex_dims[birth]].append((birth, death))
            
        rows_with_lowest_one = set()
        for k, v in pivot.items():
            rows_with_lowest_one.add(v)
        
        
        for i in range(len(self)):
            if (i not in pivot) and i not in rows_with_lowest_one:
                barcodes[self._simplex_dims[i]].append((i, float('inf')))
                
            
        return barcodes
        
        
    def __repr__(self):
        n = len(self._data)
        matrix = np.zeros((n, n))
        
        for i, col_i in enumerate(self._data):
            for j in col_i:
                matrix[j, i] = 1
                
            
        return str(matrix)
    
    def __len__(self):
        return len(self._data)
