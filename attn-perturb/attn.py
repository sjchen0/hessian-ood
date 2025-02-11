import yaml
import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np

class Config:
    """
    This is the configuration class to store the configuration of a TFModel. It is used to
    instantiate a model according to the specified arguments, defining the model architecture.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def fix_random_seed(seed, reproduce=False):
    # cudnn.enabled = True
    # cudnn.benchmark = True

    if reproduce:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ## NOTE: uncomment for CUDA >= 10.2
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        ## NOTE: uncomment for pytorch >= 1.8
        # torch.use_deterministic_algorithms(True)

    # os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = torch.manual_seed(seed)

    return rng

with open("config.yaml", "r") as file:
    config_args = yaml.safe_load(file)

config = Config(**config_args)
fix_random_seed(config.seed, reproduce=True)

A, W_OV = torch.eye(config.max_seq_len), torch.eye(config.d_model)
W_KQ = torch.eye(config.d_model)
X = torch.zeros(config.d_model, config.max_seq_len)
diag_len = min(config.d_model, config.max_seq_len)
X[range(diag_len), range(diag_len)] = torch.ones(diag_len)

X = torch.randn(config.d_model, config.max_seq_len)

print(X)

print(torch.nn.functional.softmax(X, dim=1))
print(torch.nn.functional.softmax(X@X.T, dim=1))

densities = []

# W_KQ += 1e-10 * torch.randn_like(W_KQ)

for l in range(config.num_layers):
    densities.append(torch.trace(torch.abs(A)))
    
    A = torch.nn.functional.softmax(A.T @ X.T @ W_OV.T @ W_KQ @ W_OV @ X @ A / np.sqrt(config.d_model), dim=0)
    print(A)
    # A = torch.nn.functional.softmax(X@X.T)
    print(A.shape)

print(densities)