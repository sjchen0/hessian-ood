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
from torch import autograd

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

train_dataset = torch.load("data/2_layer_64_vocab_traindata/train_dataset.pt", map_location=torch.device('cuda:0'))

mean_diff_curve = []
losses = []
epochs = list(range(7000, 10001, 1000))
# epochs = [0, 1000, 10000]
from tqdm import tqdm

for epoch in tqdm(epochs):
    print("epoch", epoch)
    checkpoint = torch.load(f"data/{dirname}/ckpt_{epoch}.pt", map_location='cuda:0')
    config.trainable_norm = False
    model = TFModel(config)
    model.load_state_dict(checkpoint, strict=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # model.eval()
    hessian = get_blkdiag_hessian(model, criterion, dataset=train_dataset, data_sample_size=5)
    torch.save(hessian, f"out/{dirname}/hessian_{epoch}.pt")