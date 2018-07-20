import chofer_torchex.pershom.pershom_backend as pershom_backend
import torch
import time
from scipy.special import binom
from itertools import combinations


from collections import Counter


point_cloud = [(0, 0), (1, 0), (0, 0.5), (1, 1.5)]
point_cloud = torch.tensor(point_cloud, device='cuda', dtype=torch.float, requires_grad=True)

def l1_norm(x, y):
    return float((x-y).abs().sum())

testee = pershom_backend.__C.VRCompCuda__PointCloud2VR_factory("l1")

args = testee(point_cloud, 2, 2)
print(args[3])
args[3].sum().backward()
print(point_cloud.grad)
















# print(c(torch.rand((10, 3), device='cuda'), 2, 0))
# print(c.filtration_values_by_dim)