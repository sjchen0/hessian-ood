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
dirname = "2_layer_64_vocab"
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

# loading model checkpoint
if config.device == "cuda":
    checkpoint = torch.load(f"data/{dirname}/ckpt_1020.pt", map_location=torch.device('cpu'))
    config.device = "cpu"

config.trainable_norm = True
model = TFModel(config)
model.load_state_dict(checkpoint, strict=False)
model.eval()
from train_module import *

criterion = (
    nn.CrossEntropyLoss(label_smoothing=0.1)
    if config.label_smoothing
    else nn.CrossEntropyLoss()
)

p = make_distr(config)
vocab = torch.arange(config.vocab_size).type(torch.LongTensor)

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

# print("completed")
# print(list(model.named_parameters()))
mean_diff_curve = []
epochs = list(range(0, 30001, 100))
for epoch in epochs:
    print(epoch)
    if config.device == "cuda":
        checkpoint = torch.load(f"data/{dirname}/ckpt_{epoch}.pt", map_location=torch.device('cpu'))
        config.device = "cpu"

    config.trainable_norm = True
    model = TFModel(config)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    diff = get_robustness_subspace(model, criterion, dataset=train_dataset, num_perturb=100, r_perturb=1e-5, data_sample_size=50, config=config, perturb_name='embed.embed.weight')

    mean_diff = [np.mean(diff[i]) * 1e10 for i in range(len(diff.keys()))]
    mean_diff_curve.append(mean_diff)