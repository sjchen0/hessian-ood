import torch.nn as nn
import torch
import warnings
import math
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from model import TFModel
import Rotary_test
import utils
import data
from torch import autograd
from tqdm import tqdm

warnings.simplefilter("ignore")
utils.fix_random_seed(42)

class Config:
    """
    This is the configuration class to store the configuration of a TFModel. It is used to
    instantiate a model according to the specified arguments, defining the model architecture.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# loading config file
dirname = "2_layer_64_vocab_traindata"
with open(f"data/{dirname}/config.json") as f:
    config_args = json.load(f)
    config_args["rotary_theta"] = 10000
config = Config(**config_args)

d_model = config.d_model
num_heads = config.num_heads
max_seq_len = config.max_seq_len
rotary_theta = config.rotary_theta
R = Rotary_test.calc_rotary_R_mat_simple(d_model, rel_dist=1)
print(R.shape)

spectrum_epoch = []
rank = 10
subspace_epoch = []

epochs = list(range(0, 10001, 1000))
for epoch in tqdm(epochs):
    print("epoch", epoch)
    checkpoint = torch.load(f"data/{dirname}/ckpt_{epoch}.pt", map_location='cuda:0')
    config.trainable_norm = True
    model = TFModel(config)
    model.load_state_dict(checkpoint, strict=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    hessian = torch.load(f"out/{dirname}/hessian_{epoch}.pt", map_location='cuda:0')[0]
    eig_val, eig_vec = torch.linalg.eigh(hessian.cpu())
    eig_mat = torch.mean(eig_vec[:,-10:-1], axis=1).reshape((config.max_seq_len, config.d_model))
    U, s, Vt = np.linalg.svd(eig_mat)

    # W = model.embed.embed.weight.detach().cpu()
    W = model.h[1].mha.W_q.weight.detach().cpu()
    U_e, s_e, V_te = np.linalg.svd(W)

    plt.figure()
    plt.plot(s, '-o')
    plt.plot(s_e, '-o')
    plt.savefig(f"plots/hessian_eig_singular_v_{epoch}.png")

    spectrum_epoch.append(eig_val.tolist())
    subspace_epoch.append(Vt @ V_te.T)

spectrum_epoch = np.array(spectrum_epoch)
plt.figure()
for i in range(spectrum_epoch.shape[1]):
    plt.plot(epochs, spectrum_epoch[:,i], '-o')
# plt.plot(epochs, np.sum(spectrum_epoch, axis=1), '-o')
plt.savefig("spectrum_epoch_no_sum.png")

for i, epoch in enumerate(epochs):
    plt.figure()
    plt.imshow(subspace_epoch[i])
    plt.colorbar()
    plt.savefig(f"plots/spectrum_subspace_alignment_{epoch}.png")