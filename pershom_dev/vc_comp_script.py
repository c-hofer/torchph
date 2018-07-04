import chofer_torchex.pershom.pershom_backend as pershom_backend
import torch

vr_persistence = pershom_backend.__C.vr_persistence

t = torch.rand(10,10)

print(vr_persistence(t))