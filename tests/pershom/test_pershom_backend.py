import pytest
import torch
import glob
import pickle
import chofer_torchex.pershom.pershom_backend as pershom_backend
from collections import Counter


class Test_find_merge_pairings:
    @pytest.mark.parametrize("device, dtype", [
        (torch.device('cuda'), torch.int32)
    ])
    def test_max_pairs(self):
        pass


    def test_result_1(self, device, dtype):
        pivots = [-1,-1, 3, 3, 3 ,5, 6, 6, 0, -1, 5, 5]
        pivots = torch.tensor(pivots, device=device, dtype=dtype)

        result = pershom_backend.find_merge_pairings(pivots, 1000)

        assert result.dtype == torch.int64

        expected_result = [(2, 3), (2, 4), 
                           (5, 10), (5, 11), 
                           (6, 7) ]

        expected_result = torch.tensor(expected_result, device=device, dtype=torch.int64)

        assert result.equal(expected_result)

        


class Test_calculate_persistence:

    @staticmethod
    def calculate_persistence_output_to_barchode_list(input):
        ret = []
        b, b_e = input    

        for dim, (b_dim, b_dim_e) in enumerate(zip(b, b_e)):
            b_dim, b_dim_e = b_dim.float(), b_dim_e.float()

            tmp = torch.empty_like(b_dim_e)
            tmp.fill_(float('inf'))
            b_dim_e = torch.cat([b_dim_e, tmp], dim=1)


            dgm = torch.cat([b_dim, b_dim_e], dim=0)
            dgm = dgm.tolist()
            dgm = Counter(((float(b), float(d)) for b, d in dgm ))

            ret.append(dgm)

        return ret

    def test_simple_1(self):
        device = torch.device('cuda')
        dtype = torch.int32

        bm = torch.empty((3, 4))
        bm.fill_(-1)
        bm[-1, 0:2] = torch.tensor([1, 0])

        row_dim = torch.tensor([0,0,1])

        max_dim = 1 

        bm = bm.to(device).type(dtype)
        row_dim = row_dim.to(device).type(dtype)       

        out = pershom_backend.calculate_persistence(bm, row_dim, max_dim, 100)

        barcodes = Test_calculate_persistence.calculate_persistence_output_to_barchode_list(out)

        assert barcodes[0] == Counter([(1.0, 2.0), (0.0, float('inf'))])


    def test_random_simplicial_complexes(self):
        device = torch.device('cuda')
        dtype = torch.int32

        for sp_path in glob.glob('test_pershom_backend_data/random_simplicial_complexes/*'):

            
            with open(sp_path, 'br') as f:
                data = pickle.load(f)

            assert len(data) == 2

            bm, row_dim, max_dim = data['calculate_persistence_args']
            expected_result = data['expected_result']

            bm, row_dim = bm.to(device).type(dtype), row_dim.to(device).type(dtype)

            result = pershom_backend.calculate_persistence(bm, row_dim, max_dim, 10000)
            result = Test_calculate_persistence.calculate_persistence_output_to_barchode_list(result)

            for dgm, dgm_exp in zip(result, expected_result):
                assert dgm == dgm_exp 








