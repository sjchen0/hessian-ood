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

train_dataset = torch.load("data/2_layer_64_vocab_traindata/train_dataset.pt", map_location=torch.device('cuda:0'))

# print("completed")
# print(list(model.named_parameters()))
mean_diff_curve = []
losses = []
epochs = list(range(0, 10001, 1000))
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
    #print("model device", model.device)
    #print("checkpoint device", checkpoint['embed.embed.weight'].device)
    #print("param device", [p.device for p in model.parameters()])

    diff = get_robustness_subspace(model, criterion, dataset=train_dataset, num_perturb=100, r_perturb=1e-3, data_sample_size=20, config=config, perturb_name='h.1.mha.W_v.weight')

    # if epoch < len(train_dataset):
    #     loss = get_loss(model, criterion, train_dataset[epoch][0])
    #     losses.append(loss.cpu())

    mean_diff = [np.mean(diff[i]) for i in range(len(diff.keys()))]
    mean_diff_curve.append(mean_diff)

mean_diff_curve = np.array(mean_diff_curve)
if False:
    for i in range(len(mean_diff_curve)):
        sharpness_mat = mean_diff_curve[i].reshape((config.d_model, config.d_model))
        plt.imshow(sharpness_mat)
        plt.savefig(f"sharpness_mat_{epochs[i]}.png")


if True:
    for i in range(mean_diff_curve.shape[1]):
        plt.plot(epochs, mean_diff_curve[:,i], '-o')
    plt.plot(epochs, np.sum(mean_diff_curve, axis=1), '-o')
    # plt.xscale('log')
    plt.savefig("subspace_sharpness.png")

# plt.figure()
# plt.plot(epochs[:-1], losses, '-o')
# plt.savefig("checkpoint_loss.png")