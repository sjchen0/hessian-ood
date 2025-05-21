# importing required libraries
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

from train_module import *

criterion = (
    nn.CrossEntropyLoss(label_smoothing=0.1)
    if config.label_smoothing
    else nn.CrossEntropyLoss()
)

p = make_distr(config)
vocab = torch.arange(config.vocab_size).type(torch.LongTensor)

'''
src_test, lens_test, starts_test, patterns = gen_simulated_data(
    distr=p,
    vocab=vocab,
    max_seq_len=config.max_seq_len,
    regime=config.regime,
    sample_size=config.sample_size_test,
    pool_size=config.pool_size,
    patterns=None,
    rep_l=config.rep_l,
    rep_h=config.rep_h,
    device=config.device,
)

train_dataset = []
for epoch in range(20):
    src, lens_train, starts_train, _ = gen_simulated_data(
        distr=p,
        vocab=vocab,
        max_seq_len=config.max_seq_len,
        regime=config.regime,
        sample_size=config.batch_size,
        pool_size=config.pool_size,
        patterns=patterns,
        rep_l=config.rep_l,
        rep_h=config.rep_h,
        device=config.device,
    )
    M = get_mask(
        src,
        lens_train,
        starts_train,
        ignore_segment=config.ignore_segment,
        ignore_burning=config.ignore_burning,
    )
    train_dataset.append((src, lens_train, starts_train, _, M))
'''

# train_dataset = torch.load("data/2_layer_64_vocab_traindata/train_dataset.pt", map_location=torch.device('cuda:0'))

# print("completed")
# print(list(model.named_parameters()))
diagonal_scores = []
sims = []
losses = []
epochs = list(range(0, 10001, 100))
# epochs = [0, 1000, 10000]
from tqdm import tqdm

for epoch in tqdm(epochs):
    print("epoch", epoch)
    checkpoint = torch.load(f"data/{dirname}/ckpt_{epoch}.pt", map_location='cuda:0')
    config.trainable_norm = True
    model = TFModel(config)
    model.load_state_dict(checkpoint, strict=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    W_ov = (model.h[0].mha.W_o.weight @ model.h[0].mha.W_v.weight ).detach()
    W_qk = (model.h[1].mha.W_q.weight.T @ model.h[1].mha.W_k.weight / math.sqrt(d_model)).detach()

    # diagonal score
    W = (W_ov+W_ov.T) @ W_qk
    diagonal_score = (torch.mean(torch.diagonal(W)) - torch.mean(W)) / torch.std(W)
    # diagonal_score = torch.trace(W) / torch.linalg.norm(W) / np.sqrt(d_model)
    diagonal_scores.append(diagonal_score.item())

    # subspace alignment
    r = 15
    U_ov, S_ov, V_ov_t = torch.linalg.svd(W_ov+W_ov.T)
    U_qk, S_qk, V_qk_t = torch.linalg.svd(W_qk)
    U_qk = U_qk[:, :r]
    V_ov = V_ov_t[:r, :].T
    _, sim, _ = torch.linalg.svd(U_qk.T @ V_ov)
    sim = torch.max(sim)
    sims.append(sim.item())

# simulate random initialization case
diagonal_scores_random = []
sims_random = []
for trial in tqdm(range(100)):
    W_ov = torch.randn(d_model, d_model) @ torch.randn(d_model, d_model)
    W_qk = torch.randn(d_model, d_model) @ torch.randn(d_model, d_model)
    W = (W_ov + W_ov.T) @ W_qk
    diagonal_score = (torch.mean(torch.diagonal(W)) - torch.mean(W)) / torch.std(W)
    diagonal_scores_random.append(diagonal_score.item())

    r = 15
    U_ov, S_ov, V_ov_t = torch.linalg.svd(W_ov + W_ov.T)
    U_qk, S_qk, V_qk_t = torch.linalg.svd(W_qk)
    U_qk = U_qk[:, :r]
    V_ov = V_ov_t[:r, :].T
    _, sim, _ = torch.linalg.svd(U_qk.T @ V_ov)
    sim = torch.max(sim)
    sims_random.append(sim.item())
diagonal_scores_random = np.mean(np.array(diagonal_scores_random))
sims_random = np.mean(np.array(sims_random))


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
ax1.plot(epochs, diagonal_scores, linewidth=3, label="trained")
ax1.set_xscale('log')
ax1.set_ylabel("diagonal score", fontsize=20)
ax1.axhline(y=diagonal_scores_random, color='r', linestyle='--', linewidth=3, label="random init")
ax1.tick_params(axis='both', labelsize=20)
ax1.legend(fontsize=16)

ax2.plot(epochs, sims, linewidth=3, label="trained")
ax2.set_xlabel("epoch", fontsize=20)
ax2.set_ylabel("subspace alignment", fontsize=20)
ax2.axhline(y=sims_random, color='r', linestyle='--', linewidth=3, label="random init")
ax2.legend(fontsize=16)
ax2.tick_params(axis='both', labelsize=20)
ax2.set_xscale('log')

plt.tight_layout()
fig.align_ylabels([ax1, ax2])
plt.savefig("dscore_and_subspace_alignment.png", dpi=300)