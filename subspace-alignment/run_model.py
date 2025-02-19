import torch.nn as nn
import torch
import fire
import os
import yaml
import numpy as np
import json

from self_attention_model import TFModel
from self_attention_utils import (
    fix_random_seed,
    create_folder,
    plot_err_curve,
)
from self_attention_config import Config
from train import train_finite, train_infinite

print(torch.__version__)

seed = 2026
fix_random_seed(seed, reproduce=True)


def make_config(
    out_dir,
    fresh_sample,
    rep_l,
    rep_h,
    regime,
    num_layers,
    mlp,
    distr,
    pos,
    pool_size,
    max_seq_len,
    ood_len_pattern,
    schedule,
    batch_size,
    num_epoch,
    plot_attn_every_epoch,
):
    ### set up default hyperparameters in ./config/config.yaml
    with open("./config/config.yaml", "r") as file:
        config_args = yaml.safe_load(file)
    # Create config object
    config = Config(**config_args)

    config.out_dir = out_dir
    config.fresh_sample = fresh_sample
    config.regime = regime
    config.num_layers = num_layers
    config.mlp = mlp
    config.pos = pos  # 'rotary' or None
    config.pool_size = pool_size
    config.num_epoch = num_epoch
    config.schedule = schedule
    config.batch_size = batch_size
    config.max_seq_len = max_seq_len
    config.rep_l, config.rep_h = rep_l, rep_h
    config.plot_attn_every_epoch = plot_attn_every_epoch
    config.ood_len_pattern = ood_len_pattern
    config.distr = distr  # None or 'two-level' or 'two-level-4'

    return config


def create_save_dir_name(config):
    str_regime = "varied" if config.regime == "varied repetition" else ""
    str_distr = "_"+config.distr if config.distr is not None else ""
    str_pool_size = f"_pool_{config.pool_size}" if config.pool_size is not None else ""
    str_linear_attn = "_linearAttn" if config.linear_attn else ""
    str_dmodel = "_dim_"+str(config.d_model)
    str_sample_size = (
        f"_sample_{config.sample_size}" if not config.fresh_sample else "_fresh_sample"
    )
    str_epoch = "_epoch_" + str(config.num_epoch)
    str_lr = "_lr_" + str(config.lr)[2:]
    str_schedule = f"_schedule_{config.schedule}"
    str_batch_size = f"_batch_size_{config.batch_size}"
    str_rel_pos = "_" + config.pos if config.pos is not None else ""
    str_num_layer = f"_layer_{config.num_layers}" if config.num_layers != 2 else ""


    save_dir = os.path.join(
        config.out_dir,
        str_regime
        + str_distr
        + str_pool_size
        + str_linear_attn
        + str_num_layer
        + str_dmodel
        + str_rel_pos
        + str_sample_size
        + str_schedule
        + str_batch_size
        + str_epoch
        + str_lr,
    )
    return save_dir


def main(
    out_dir="out",
    fresh_sample=True,
    rep_l=12,
    rep_h=20,
    regime="varied repetition",
    num_layers=2,
    mlp=False,
    distr="two-level-4",
    pos="rotary",
    schedule="constant",
    pool_size=None,
    max_seq_len=64,
    ood_len_pattern=25,
    batch_size=64,
    num_epoch=25000,
    print_output=False,
    plot_attn_every_epoch=500,

):
    config = make_config(
        out_dir=out_dir,
        fresh_sample=fresh_sample,
        rep_l=rep_l,
        rep_h=rep_h,
        regime=regime,
        num_layers=num_layers,
        mlp=mlp,
        distr=distr,
        pos=pos,
        pool_size=pool_size,
        max_seq_len=max_seq_len,
        ood_len_pattern=ood_len_pattern,
        batch_size=batch_size,
        num_epoch=num_epoch,
        schedule=schedule,
        plot_attn_every_epoch=plot_attn_every_epoch,
    )

    save_dir = create_save_dir_name(config)
    save_attn_dir = os.path.join(save_dir, "figures")
    create_folder(save_dir)
    create_folder(save_attn_dir)
    print(config.__dict__)
    json.dump(config.__dict__, open(os.path.join(save_dir, "config.json"), "w"))

    model = TFModel(config).to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config.wd,
    )

    if schedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config.num_epoch // 4
        )
        anneal = True
    elif schedule == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        anneal = False
    elif schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config.num_epoch
        )
        anneal = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if config.fresh_sample:
        model, err_arr, err_arr_json = train_infinite(
            model=model,
            optimizer=optimizer,
            config=config,
            print_output=print_output,
            scheduler=scheduler,
            anneal=anneal,
            save_plot_dir=save_dir,
            plot_attn_every_epoch=plot_attn_every_epoch,
        )

    else:
        model, err_arr, err_arr_json = train_finite(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            print_output=print_output,
            scheduler=scheduler,
            anneal=anneal,
            save_plot_dir=save_dir,
            plot_attn_every_epoch=plot_attn_every_epoch,
        )

    ## save model
    save_path = os.path.join(save_dir, "ckpt.pt")
    torch.save(model.state_dict(), save_path)
    model.eval()

    plot_err_curve(
        err_arr,
        fig_name="train_test_curves",
        save_dir=save_dir,
        plot_ood=True,
        plot_train=not fresh_sample,
        log_training_time=fresh_sample,
    )

    json.dump(
        err_arr_json,
        open(
            os.path.join(save_dir, "err_arr.json"),
            "w",
        ),
    )


if __name__ == "__main__":
    fire.Fire(main)
