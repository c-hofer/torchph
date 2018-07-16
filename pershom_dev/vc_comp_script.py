import chofer_torchex.pershom.pershom_backend as pershom_backend
import torch
import time
from scipy.special import binom


 #torch.tensor([[-0.6690,  1.5059], [ 0.4220,  1.2434], [-0.3436, -0.0053], [-0.1569,  0.0627]], device='cuda', requires_grad=True).float()


point_cloud = torch.randn(5,3, device='cuda', requires_grad=True).float()
# point_cloud = torch.tensor([(0, 0), (1, 0), (0, 0.5), (1, 1.5)], device='cuda', requires_grad=True)

# loss = x.sum()
# loss.backward()
# print(point_cloud.grad)


print(point_cloud)
# pershom_backend.__C.CalcPersCuda__my_test_f(point_cloud).sum().backward(); 

# time_start = time.time()
try:
    r = pershom_backend.__C.VRCompCuda__vr_l1_persistence(point_cloud, 0, 0)

except Exception as ex: 
    print("=== Error ===")
    print(ex)
    exit()

non_essentials = r[0]
essentials = r[1]

print(len(non_essentials), len(essentials))

print("=== non-essentials ===")
for x in non_essentials: print(x)

print("=== essentials ===")
for x in essentials: print(x) 

print("=== grad ===")
loss = non_essentials[0].sum()
loss.backward()
print(point_cloud.grad)

# ba = r[0]
# simp_dim = r[2]
# filt_val = r[3]

# dim = 3
# for simp_id, boundaries in enumerate(ba):
#     simp_filt_val = filt_val[simp_id + point_cloud.size(0)]

#     for boundary in boundaries.tolist():
#         if boundary == -1: continue         
#         cond = True 
#         cond = cond and (simp_dim[simp_id + point_cloud.size(0)] - 1 == simp_dim[boundary])
#         cond = cond and filt_val[boundary] <= simp_filt_val

#         if not cond:
#             print("{}, {}".format(simp_id, boundary))
#             print(ba[simp_id])
#             print(simp_dim[simp_id + point_cloud.size(0)], simp_dim[boundary])

#             raise Exception()




    
# print(print(r[2][int(5+binom(5,2)):int(5+binom(5,2))+1000]))


# loss = t.sum()
# loss.backward()
# 
# for i, r in enumerate(ba):
#     x = point_cloud[r[0]-point_cloud.size(0)]
#     y = point_cloud[r[1]-point_cloud.size(0)]

#     assert torch.equal((x-y).abs().sum(), t[i]) 


# for row_i in range(t.size(0)):
#     for col_i in range(t.size(1)):
#         assert int(t[row_i, col_i]) == binom(col_i, row_i+1)
# print(time.time() - time_start)
