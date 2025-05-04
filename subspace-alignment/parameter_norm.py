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

W_ov_norms = []
W_embed_norms = []
W_qk_norms = []
W_ov_1_norms = []
W_qk_1_norms = []

epochs = list(range(0, 10001, 100))
for epoch in tqdm(epochs):
    # print("epoch", epoch)
    checkpoint = torch.load(f"data/{dirname}/ckpt_{epoch}.pt", map_location='cpu')
    config.trainable_norm = True
    model = TFModel(config)
    model.load_state_dict(checkpoint, strict=False)
    device = torch.device(config.device)
    model.to(device)
    model.eval()

    W_embed = model.embed.embed.weight.detach().cpu()
    W_embed_norms.append(torch.linalg.norm(W_embed).item())

    W_ov = model.h[0].mha.W_o.weight.detach().cpu() @ model.h[0].mha.W_v.weight.T.detach().cpu()
    W_ov_norms.append(torch.linalg.norm(W_ov).item())

    W_qk = model.h[0].mha.W_q.weight.detach().cpu() @ model.h[0].mha.W_k.weight.T.detach().cpu()
    W_qk_norms.append(torch.linalg.norm(W_qk).item())

    W_ov_1 = model.h[1].mha.W_o.weight.detach().cpu() @ model.h[1].mha.W_v.weight.T.detach().cpu()
    W_ov_1_norms.append(torch.linalg.norm(W_ov_1).item())

    W_qk_1 = model.h[1].mha.W_q.weight.detach().cpu() @ model.h[1].mha.W_k.weight.T.detach().cpu()
    W_qk_1_norms.append(torch.linalg.norm(W_qk_1).item())

plt.figure()
# plt.plot(epochs, W_embed_norms, '-o', label='$W_{embed}$')
plt.plot(epochs, W_ov_norms, '-o', label='$W_{OV}^1$')
plt.plot(epochs, W_qk_norms, '-o', label='$W_{QK}^1$')
plt.plot(epochs, W_ov_1_norms, '-o', label='$W_{OV}^2$')
plt.plot(epochs, W_qk_1_norms, '-o', label='$W_{QK}^2$')
plt.legend()
plt.savefig("plots/W_ov_norms.png", dpi=300)