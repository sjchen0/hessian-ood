import torch
from functorch import hessian
from torch.nn.utils import _stateless
import time

# Create model
model = torch.nn.Sequential(torch.nn.Linear(5, 100), torch.nn.Tanh(), torch.nn.Linear(100, 5))
num_param = sum(p.numel() for p in model.parameters())
names = list(n for n, _ in model.named_parameters())

# Create random dataset
x = torch.rand((1000,5))
y = torch.rand((1000,5))

# Define loss function
def loss(params):
    y_hat = _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, x)
    return ((y_hat - y)**2).mean()

# Calculate Hessian
hessian_func = hessian(loss)

start = time.time()

H = hessian_func(tuple(model.parameters()))

import numpy as np
np.linalg.norm((H[2][0].permute(2,3,0,1) - H[0][2]).detach().numpy())

def reduce_shape(M, shape_1, shape_2):
    '''
    Used to flatten a high-dimensional Hessian tensor into a Hessian matrix and ensure symmetry.
    '''
    if len(shape_1) not in (1,2) or len(shape_2) not in (1,2):
        raise NotImplementedError
    if len(shape_1) == 2:
        M = M.flatten(start_dim=0, end_dim=1)
    if len(shape_2) == 2:
        M = M.flatten(start_dim=1, end_dim=2)
    return M

shapes = [p.shape for p in model.parameters()]
H = torch.cat([torch.cat([reduce_shape(H[i][j], shapes[i], shapes[j]) for j in range(len(H))], axis=1) for i in range(len(H))], axis=0)


H = torch.cat([torch.cat([e.flatten() for e in Hpart]) for Hpart in H]) # flatten
H = H.reshape(num_param, num_param)

print(time.time() - start)

