import pytest
import torch
import glob
import pickle
import chofer_torchex.pershom.pershom_backend as pershom_backend
from collections import Counter


class Test_find_merge_pairings:
    @pytest.mark.parametrize("device, dtype", [
        (torch.device('cuda'), torch.int64)
    ])
    def test_return_value_dtype(self, device, dtype):
        pivots = torch.tensor([1, 1], device=device, dtype=dtype)

        result = pershom_backend.find_merge_pairings(pivots)

        assert result.dtype == torch.int64

    @pytest.mark.parametrize("device, dtype", [
        (torch.device('cuda'), torch.int64)
    ])
    def test_parameter_max_pairs(self, device, dtype):
        pivots = torch.tensor([1]*1000, device=device, dtype=dtype).unsqueeze(1)

        result = pershom_backend.find_merge_pairings(pivots, max_pairs=100)

        assert result.size(0) == 100

    @pytest.mark.parametrize("device, dtype", [
        (torch.device('cuda'), torch.int64)
    ])
    def test_no_merge_pairs(self, device, dtype):
        pivots = torch.tensor(list(range(100)), device=device, dtype=dtype).unsqueeze(1)
        assert pershom_backend.find_merge_pairings(pivots).numel() == 0

    @pytest.mark.parametrize("device, dtype", [
        (torch.device('cuda'), torch.int64)
    ])
    def test_result_1(self, device, dtype):
        pivots = [6, 3, 3, 3, 5, 6, 6, 0, 5, 5]
        pivots = torch.tensor(pivots, device=device, dtype=dtype).unsqueeze(1)

        result = pershom_backend.find_merge_pairings(pivots)

        assert result.dtype == torch.int64

        expected_result = set([(0, 5), (0, 6),
                               (1, 2), (1, 3),
                               (4, 8), (4, 9)])

        result = set(tuple(x) for x in result.tolist())

        assert result == (expected_result)

    @pytest.mark.parametrize("device, dtype", [
        (torch.device('cuda'), torch.int64)
    ])
    def test_result_2(self, device, dtype):
        pivots = sum([100*[i] for i in range(100)], [])
        pivots = torch.tensor(pivots, device=device, dtype=dtype).unsqueeze(1)

        expected_result = torch.tensor([(int(i/100) * 100, i) for i in range(100 * 100) if i % 100 != 0])
        expected_result = expected_result.long().to('cuda')

        result = pershom_backend.find_merge_pairings(pivots)

        assert expected_result.equal(result)


class Test_calculate_persistence:

    @staticmethod
    def calculate_persistence_output_to_barcode_list(input):
        ret = []
        b, b_e = input

        for dim, (b_dim, b_dim_e) in enumerate(zip(b, b_e)):
            b_dim, b_dim_e = b_dim.float(), b_dim_e.float()

            tmp = torch.empty_like(b_dim_e)
            tmp.fill_(float('inf'))
            b_dim_e = torch.cat([b_dim_e, tmp], dim=1)

            dgm = torch.cat([b_dim, b_dim_e], dim=0)
            dgm = dgm.tolist()
            dgm = Counter(((float(b), float(d)) for b, d in dgm))

            ret.append(dgm)

        return ret

    @pytest.mark.parametrize("device, dtype", [
        (torch.device('cuda'), torch.int64)
    ])
    def test_empty_input(self, device, dtype):
        ba = torch.empty([0], device=device, dtype=dtype)
        ba_row_i_to_bm_col_i = ba
        simplex_dimension = torch.zeros(10, device=device, dtype=dtype)

        not_ess, ess = pershom_backend.calculate_persistence(
            ba,
            ba_row_i_to_bm_col_i,
            simplex_dimension,
            2,
            -1)

        for pairings in not_ess:
            assert len(pairings) == 0

        assert len(ess[0]) == 10

        for birth_i in ess[1:]:
            assert len(birth_i) == 0

    def test_simple_1(self):
        device = torch.device('cuda')
        dtype = torch.int64

        ba = torch.empty((1, 4))
        ba.fill_(-1)
        ba[0, 0:2] = torch.tensor([1, 0])
        ind_not_reduced = torch.tensor([2])

        row_dim = torch.tensor([0, 0, 1])

        max_dim_to_read_of_reduced_ba = 1

        ba = ba.to(device).type(dtype)
        ind_not_reduced = ind_not_reduced.to(device).long()
        row_dim = row_dim.to(device).type(dtype)

        out = pershom_backend.calculate_persistence(
            ba,
            ind_not_reduced,
            row_dim,
            max_dim_to_read_of_reduced_ba,
            100)

        barcodes = Test_calculate_persistence.calculate_persistence_output_to_barcode_list(out)

        assert barcodes[0] == Counter([(1.0, 2.0), (0.0, float('inf'))])

    def test_random_simplicial_complexes(self):
        device = torch.device('cuda')
        dtype = torch.int64

        for sp_path in glob.glob('test_pershom_backend_data/random_simplicial_complexes/*'):

            with open(sp_path, 'br') as f:
                data = pickle.load(f)

            assert len(data) == 2

            ba, ind_not_reduced, row_dim, max_dim_to_read_of_reduced_ba = data['calculate_persistence_args']
            expected_result = data['expected_result']

            ba, ind_not_reduced, row_dim = ba.to(device).type(dtype), ind_not_reduced.to(device).long(), row_dim.to(device).type(dtype)

            result = pershom_backend.calculate_persistence(
                ba,
                ind_not_reduced,
                row_dim,
                max_dim_to_read_of_reduced_ba,
                10000)
            result = Test_calculate_persistence.calculate_persistence_output_to_barcode_list(result)

            for dgm, dgm_exp in zip(result, expected_result):
                assert dgm == dgm_exp
