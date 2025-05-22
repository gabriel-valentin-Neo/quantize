import torch
import neoQuant




tensorA = torch.randn(32, 32, dtype=torch.float16, device='cuda')

tensorA = tensorA * 1000


print(torch.max(tensorA, dim=1)[0])


cute_final = neoQuant.scales(tensorA)
#cute_tensor = cute.Tensor

#print(tensor)

#print()
#neoQuant.scales(tensorA, scales_tensor)

#print(scales_tensor)
