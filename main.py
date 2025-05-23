import json
import os

import fire
import torch
import yaml
from model import TFModel
from train import train_finite, train_infinite
from utils import create_folder, fix_random_seed, plot_err_curve, plot_sharpness_curve
import numpy as np

print(torch.__version__)


class Config:
    """
    This is the configuration class to store the configuration of a TFModel. It is used to
    instantiate a model according to the specified arguments, defining the model architecture.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def make_scheduler(optimizer, config):
    if config.schedule == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    elif config.schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config.num_epoch
        )
    return scheduler


def main(config_file = "./config.yaml", **kwargs):
    with open(config_file, "r") as file:
        config_args = yaml.safe_load(file)
    for k, v in kwargs.items():
        if k not in config_args:
            print(f"Warning: {k} is not supported!")
        if v != config_args[k]:
            print(f"{k} is overloaded from {config_args[k]} to {v}")
            config_args[k] = v
    config = Config(**config_args)

    fix_random_seed(config.seed, reproduce=True)

    create_folder(config.out_dir)
    create_folder(os.path.join(config.out_dir, "figures"))

    print(config.__dict__)
    json.dump(config.__dict__, open(os.path.join(config.out_dir, "config.json"), "w"))

    model = TFModel(config).to(config.device)

    ## save init model
    out_path = os.path.join(config.out_dir, "ckpt_0.pt")
    torch.save(model.state_dict(), out_path)

    use_sam = False
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=config.wd,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=config.wd,
        )
    elif config.optimizer == "sam":
        use_sam = True
        from sam import SAM
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(
            model.parameters(), 
            base_optimizer,
            rho=config.sam_rho,
            lr=config.lr,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=config.wd,
        )

    scheduler = make_scheduler(optimizer, config)

    if config.fresh_sample:
        model, err_arr, err_arr_json = train_infinite(
            model=model,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
            use_sam = use_sam,
        )

    else:
        model, err_arr, err_arr_json = train_finite(
            model=model,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    model.eval()

    ## save model
    out_path = os.path.join(config.out_dir, "ckpt.pt")
    torch.save(model.state_dict(), out_path)

    model.eval()
    plot_err_curve(
        err_arr,
        fig_name="train_test_curves",
        save_dir=config.out_dir,
        plot_ood=True,
        plot_train=not config.fresh_sample,
        log_training_time=config.fresh_sample,
    )
    '''
    plot_sharpness_curve(
        sharpness_arr,
        fig_name="sharpness_curve",
        save_dir=config.out_dir
    )
    for k in diff_by_blk_summary.keys():
        plot_sharpness_curve(
            np.array(diff_by_blk_summary[k]),
            fig_name=f"diff_by_blk_{k}",
            save_dir=config.out_dir
        )
    '''

    json.dump(
        err_arr_json,
        open(
            os.path.join(config.out_dir, "err_arr.json"),
            "w",
        ),
    )


if __name__ == "__main__":
    fire.Fire(main)
