# importing required libraries
import os
import warnings
import psutil
import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from tqdm import tqdm
from data import gen_repetition_data, gen_simple_data, gen_mod_add_data
from utils import (
    mask_get_along_axis,
    mask_get_given_starts,
    plot_err_over_pos,
    plots_maker,
    plot_blk_spectrum
)
from hessian import *
warnings.simplefilter("ignore")


def get_mask(src, lens, starts=None, ignore_segment=0, ignore_burning=0):
    M = torch.ones_like(src)
    if lens is not None and starts is None:
        M = torch.Tensor(
            mask_get_along_axis(
                src.shape,
                lens,
                ignore_segment=ignore_segment,
                ignore_burning=ignore_burning,
            )
        )
    elif lens is not None and starts is not None:
        M = torch.Tensor(
            mask_get_given_starts(
                src.shape,
                lens,
                starts,
                ignore_segment=ignore_segment,
                ignore_burning=ignore_burning,
            )
        )
    return M


def get_loss(model, criterion, src):
    output = model(src)
    vocab_size = output.size(-1)
    loss = criterion(
        output[:, :-1].contiguous().view(-1, vocab_size),
        src[:, 1:].contiguous().view(-1),
    )
    return loss

@torch.no_grad()
def loss_err(model, criterion, src, mask):
    model.eval()
    output = model(src)
    vocab_size = output.size(-1)
    loss = criterion(
        output[:, :-1].contiguous().view(-1, vocab_size),
        src[:, 1:].contiguous().view(-1),
    )

    tmp = output.argmax(dim=2)[:, :-1] == src[:, 1:]
    err = 1 - torch.sum(tmp.cpu() * mask[:, :-1], dtype=torch.float) / torch.sum(mask)
    return loss, err


def gen_simulated_data(
    distr,
    vocab,
    max_seq_len,
    sample_size,
    regime,
    pool_size,
    patterns,
    rep_l,
    rep_h,
    device,
):
    if regime == "simple repetition":
        src, lens = gen_simple_data(
            vocab,
            max_seq_len,
            sample_size,
            return_lens=True,
            rep_l=rep_l,
            rep_h=rep_h,
        )

        return src.to(device), lens, None, None

    elif regime == "varied repetition":
        src, lens, starts, patterns = gen_repetition_data(
            vocab,
            max_seq_len,
            sample_size,
            distr=distr,
            pattern_pool_size=pool_size,
            patterns=patterns,
            return_lens=True,
            rep_l=rep_l,
            rep_h=rep_h,
        )

        return src.to(device), lens, starts, patterns
    
    elif regime == "modular addition":
        src, lens, starts, patterns = gen_mod_add_data(
            vocab,
            max_seq_len,
            sample_size,
            distr=distr,
            pattern_pool_size=pool_size,
            patterns=patterns,
            return_lens=True,
            rep_l=rep_l,
            rep_h=rep_h,
        )

        return src.to(device), lens, starts, patterns


def make_distr(config):
    if config.distr == "two-level":
        p = np.concatenate(
            (
                np.array([1 / 8] * 4),
                np.array([1 / (2 * (config.vocab_size - 4))] * (config.vocab_size - 4)),
            )
        )
        # np.random.shuffle(p)
        p = torch.Tensor(p)
    elif config.distr == "two-level-3":  # NOT USED for now, may change later
        p = np.concatenate(
            (
                np.array([1 / 8] * 4),
                np.array([1 / (2 * (config.vocab_size - 4))] * (config.vocab_size - 4)),
            )
        )
        # np.random.shuffle(p)
        p = torch.Tensor(p)
    elif config.distr == "zipf":
        # https://en.wikipedia.org/wiki/Zipf%27s_law
        p = np.array([1 / (i + 2.7) for i in range(1, config.vocab_size + 1)])
        p = p / np.sum(p)
        # np.random.shuffle(p)
        p = torch.Tensor(p)
    elif config.distr == "unif":
        p = None
    else:
        raise ValueError(f"distr {config.distr} is not supported!")

    return p


#####################################################
##################### Training #########################
#####################################################


def train_infinite(
    model,
    config,
    optimizer,
    scheduler,
):
    num_epoch = config.num_epoch
    batch_size = config.batch_size
    vocab = torch.arange(config.vocab_size).type(torch.LongTensor)
    p = make_distr(config)

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

    src_test_ood, lens_test_ood, starts_test_ood, _ = gen_simulated_data(
        distr=None,
        vocab=vocab,
        max_seq_len=config.max_seq_len,
        regime=config.regime,
        sample_size=config.sample_size_test,
        pool_size=None,
        patterns=None,
        rep_l=config.ood_len_pattern,
        rep_h=config.ood_len_pattern + 1,
        device=config.device,
    )

    M_test = get_mask(
        src_test,
        lens_test,
        starts_test,
        ignore_segment=config.ignore_segment,
        ignore_burning=config.ignore_burning,
    )
    M_test_ood = get_mask(
        src_test_ood,
        lens_test_ood,
        starts_test_ood,
        ignore_segment=config.ignore_segment,
        ignore_burning=config.ignore_burning,
    )

    torch.save(
        [src_test, lens_test, starts_test], os.path.join(config.out_dir, "test.pth")
    )
    torch.save(
        [src_test_ood, lens_test_ood, starts_test_ood],
        os.path.join(config.out_dir, "test_ood.pth"),
    )

    err_arr = np.zeros((num_epoch, 6))
    sharpness_arr = np.zeros((num_epoch,))
    trial_sharpness_arr = np.zeros((num_epoch, 2000))
    diff_by_blk_summary = dict()

    err_arr_json = []
    criterion = (
        nn.CrossEntropyLoss(label_smoothing=0.1)
        if config.label_smoothing
        else nn.CrossEntropyLoss()
    )


    train_dataset = []
    for epoch in range(num_epoch):
        src, lens_train, starts_train, _ = gen_simulated_data(
            distr=p,
            vocab=vocab,
            max_seq_len=config.max_seq_len,
            regime=config.regime,
            sample_size=batch_size,
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


    for epoch in tqdm(range(num_epoch)):
        model.train()

        optimizer.zero_grad()

        src, lens_train, starts_train, _, M = train_dataset[epoch]

        '''
        src, lens_train, starts_train, _ = gen_simulated_data(
            distr=p,
            vocab=vocab,
            max_seq_len=config.max_seq_len,
            regime=config.regime,
            sample_size=batch_size,
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
        '''
        loss = get_loss(model, criterion, src)
        loss.backward()
        optimizer.step()
        

        with torch.no_grad():
            model.eval()  # useful if dropout or batchnorm etc is turned on
            loss_train, train_err = loss_err(model, criterion, src, M)
            loss_test, test_err = loss_err(model, criterion, src_test, M_test)
            loss_test_ood, test_err_ood = loss_err(
                model, criterion, src_test_ood, M_test_ood
            )
            if False: # compute full Hessian
                hessian_train = get_hessian(model, criterion, src=src, dataset=train_dataset)
                sharpness = torch.trace(hessian_train)
                print(sharpness)


        if False: # compute block-diagonal Hessian
            if (epoch % 100 == 0 and 1000 <= epoch <= 2000) or epoch in [0,700]:
                blkdiag_hessian_train = get_blkdiag_hessian(model, criterion, src=src, dataset=train_dataset)
                avg_sharpness = sum([torch.trace(h) for h in blkdiag_hessian_train])
                blk_spectrums = [torch.linalg.eigh(h)[0] for h in blkdiag_hessian_train]
                # spectrum = torch.concat(blk_spectrums)
                
                # plot spectrum
                parameter_names = [name for name, _ in model.named_parameters()]
                plot_blk_spectrum(
                   blk_spectrums, 
                   parameter_names, 
                   fig_name=f"spectrum_epoch_{epoch}", 
                   save_dir=config.out_dir
                )

                import matplotlib.pyplot as plt
                spectrum = torch.concat(blk_spectrums)
                plt.figure()
                plt.hist(spectrum, 100)
                plt.yscale('log')
                plt.savefig(os.path.join(config.out_dir, f"spectrum_hist_epoch_{epoch}"))

        if False: # directly compute Hessian trace
            if epoch % 1000 == 0: # (epoch % 100 == 0 and 1000 <= epoch <= 2000) or epoch in [0,700]:
                sharpness_trace = get_trace_hessian(model, criterion, src=src, dataset=train_dataset)
                avg_sharpness = sum(sharpness_trace) / len(sharpness_trace)
                
                # scale the sharpness by model weight
                # avg_sharpness *= sum([torch.norm(p).item()**2 for p in model.parameters()])

        if False:
            if epoch % config.sharpness_step == 0:
                avg_sharpness, diff_by_blk = get_robustness_blk(model, criterion, src=src, dataset=train_dataset, num_perturb=100, r_perturb=1e-3, data_sample_size=20, config=config)

        if True:
            if epoch % config.sharpness_step == 0:
                diff = get_robustness(model, criterion, src=src, dataset=train_dataset, num_perturb=100, r_perturb=1e-3, data_sample_size=20, config=config)
                avg_sharpness = sum(diff) / len(diff)

        '''
        if len(diff_by_blk_summary) == 0:
            for k in diff_by_blk.keys():
                diff_by_blk_summary[k] = [diff_by_blk[k].item()]
        else:
            for k in diff_by_blk.keys():
                diff_by_blk_summary[k].append(diff_by_blk[k].item())
        '''

        sharpness_arr[epoch] = avg_sharpness
        trial_sharpness_arr[epoch] = np.array([d.item() for d in diff])

        err_arr[epoch, :] = [
            loss_train.item(),
            train_err.item(),
            loss_test.item(),
            test_err.item(),
            loss_test_ood.item(),
            test_err_ood.item(),
        ]

        err_arr_json += [
            {
                "epoch": epoch,
                "loss_train": loss_train.item(),
                "err_train": train_err.item(),
                "loss_test": loss_test.item(),
                "err_test": test_err.item(),
                "loss_ood": loss_test_ood.item(),
                "err_ood": test_err_ood.item(),
            }
        ]

        scheduler.step()

        if epoch % config.plot_attn_every_epoch == 0 and err_arr[epoch, 5] > 0.05:
            plots_maker(
                model,
                config,
                [src, src_test, src_test_ood],
                epoch=epoch,
                lens=[lens_train, lens_test, lens_test_ood],
                starts=[starts_train, starts_test, starts_test_ood],
                save_dir=os.path.join(config.out_dir, "figures"),
            )

            if config.print_output:
                print(
                    f"----> Epoch: {epoch+1:>5}, Train Loss: {loss.item():.3f}, Test Error: {err_arr[epoch,3]:.3f}, OOD Error: {err_arr[epoch,5]:.3f}"
                )

        if (1 + epoch) % (config.num_epoch // config.n_save) == 0 or (
            config.up_to_first_save
            and (1 + epoch)
            in [
                np.power(2, k)
                for k in range(int(np.log2(config.num_epoch // config.n_save)))
            ]
        ):
            out_path = os.path.join(config.out_dir, f"ckpt_{epoch + 1}.pt")
            torch.save(model.state_dict(), out_path)

    lens = [lens_train, lens_test, lens_test_ood]
    _ = plot_err_over_pos(
        model,
        [src, src_test, src_test_ood],
        config.vocab_size,
        "err_over_pos",
        lens=lens,
        starts=[starts_train, starts_test, starts_test_ood],
        src_labels=["train", "test", "ood"],
        save_dir=config.out_dir,
    )

    np.save("out/trial_diff.npy", trial_sharpness_arr)

    return model, err_arr, sharpness_arr, diff_by_blk_summary, err_arr_json


def train_finite(
    model,
    optimizer,
    criterion=nn.CrossEntropyLoss(),
    config=None,
    print_output=False,
    scheduler=None,
    anneal=False,
    save_plot_dir=None,
    plot_attn_every_epoch=10,
    masking_config=[1, 4],
):
    raise ValueError("Finite setting is out of date.")

    num_epoch = config.num_epoch
    batch_size = config.batch_size
    epoch_change = config.num_epoch // 4
    vocab = torch.arange(config.vocab_size).type(torch.LongTensor)
    if config.distr == "two-level":
        p = np.concatenate(
            (
                np.array([1 / 8] * 4),
                np.array([1 / (2 * (config.vocab_size - 4))] * (config.vocab_size - 4)),
            )
        )
        np.random.shuffle(p)
        p = torch.Tensor(p)
    else:
        p = None

    src, lens_train, starts_train, patterns = gen_simulated_data(
        distr=p,
        vocab=vocab,
        max_seq_len=config.max_seq_len,
        regime=config.regime,
        sample_size=config.sample_size,
        pool_size=config.pool_size,
        patterns=None,
        rep_l=config.rep_l,
        rep_h=config.rep_h,
        device=config.device,
    )
    M = get_mask(
        src,
        lens_train,
        starts_train,
        ignore_segment=masking_config[0],
        ignore_burning=masking_config[1],
    )

    src_test, lens_test, starts_test, _ = gen_simulated_data(
        distr=p,
        vocab=vocab,
        max_seq_len=config.max_seq_len,
        regime=config.regime,
        sample_size=config.sample_size_test,
        pool_size=config.pool_size,
        rep_l=config.rep_l,
        rep_h=config.rep_h,
        patterns=patterns,
        device=config.device,
    )

    src_test_ood, lens_test_ood, starts_test_ood, _ = gen_simulated_data(
        distr=None,
        vocab=vocab,
        max_seq_len=config.max_seq_len,
        regime=config.regime,
        pool_size=None,
        patterns=None,
        sample_size=config.sample_size_test,
        rep_l=config.ood_len_pattern,
        rep_h=config.ood_len_pattern + 1,
        device=config.device,
    )

    M_test = get_mask(
        src_test,
        lens_test,
        starts_test,
        ignore_segment=masking_config[0],
        ignore_burning=masking_config[1],
    )
    M_test_ood = get_mask(
        src_test_ood,
        lens_test_ood,
        starts_test_ood,
        ignore_segment=masking_config[0],
        ignore_burning=masking_config[1],
    )

    err_arr = np.zeros((num_epoch, 6))
    err_arr_json = []
    for epoch in tqdm(range(num_epoch)):
        model.train()

        perm = np.arange(config.sample_size, dtype=int)
        np.random.shuffle(perm)
        for batch_idx in range(config.sample_size // batch_size):
            indices = perm[
                range((batch_size * batch_idx), (batch_size * batch_idx + batch_size))
            ]
            optimizer.zero_grad()
            loss = loss_err(model, criterion, src[indices], M, return_err=False)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()  # useful if dropout or batchnorm etc is turned on

            loss_train, train_err = loss_err(model, criterion, src, M, return_err=True)

            loss_test, test_err = loss_err(
                model, criterion, src_test, M_test, return_err=True
            )
            loss_test_ood, test_err_ood = loss_err(
                model, criterion, src_test_ood, M_test_ood, return_err=True
            )

            err_arr[epoch, :] = [
                loss_train.item(),
                train_err.item(),
                loss_test.item(),
                test_err.item(),
                loss_test_ood.item(),
                test_err_ood.item(),
            ]

            err_arr_json += [
                {
                    "epoch": epoch,
                    "loss_train": loss_train.item(),
                    "err_train": train_err.item(),
                    "loss_test": loss_test.item(),
                    "err_test": test_err.item(),
                    "loss_ood": loss_test_ood.item(),
                    "err_ood": test_err_ood.item(),
                }
            ]

        scheduler.step()
        if anneal and (epoch + 1) % epoch_change == 0:  # restart
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, epoch_change
            )
        if print_output:
            print(
                f"----> Epoch: {epoch+1:>5}, Train Loss: {loss.item():.3f}, Train Error: {err_arr[epoch,1]:.3f}, Test Error: {err_arr[epoch,3]:.3f}, OOD Error: {err_arr[epoch,5]:.3f}, lr: {scheduler.get_last_lr()[0]:.5f}"
            )

        if (
            save_plot_dir is not None
            and epoch % plot_attn_every_epoch == 0
            and err_arr[epoch, 5] > 0.5
        ):
            plots_maker(
                model,
                config,
                [src, src_test, src_test_ood],
                epoch=epoch,
                lens=[lens_train, lens_test, lens_test_ood],
                starts=[starts_train, starts_test, starts_test_ood],
                save_dir=os.path.join(save_plot_dir, "figures"),
            )

    lens = [lens_train, lens_test, lens_test_ood]
    _ = plot_err_over_pos(
        model,
        [src, src_test, src_test_ood],
        config.vocab_size,
        "err_over_pos",
        lens=lens,
        starts=[starts_train, starts_test, starts_test_ood],
        src_labels=["train", "test", "ood"],
        save_dir=save_plot_dir,
    )

    return model, err_arr, err_arr_json
