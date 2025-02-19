import torch
import numpy as np
import data
import utils

def get_robustness(model, criterion, **kwargs):
    dataset = kwargs['dataset']
    num_perturb = kwargs['num_perturb']
    r_perturb = kwargs['r_perturb']
    config = kwargs['config']
    data_sample_size = kwargs['data_sample_size']
    perturb_name = kwargs['perturb_name']
    indices = np.random.permutation(np.arange(len(dataset)))[:data_sample_size]
    names = list(n for n, _ in model.named_parameters())

    def batch_loss(params, src):
        output = torch.func.functional_call(model, {n: p.detach() for n, p in zip(names, params)}, src)
        vocab_size = output.size(-1)
        loss = criterion(
            output[:, :-1].contiguous().view(-1, vocab_size),
            src[:, 1:].contiguous().view(-1),
        )
        return loss

    diff = []

    for idx in indices:
        params = list(model.parameters())
        src = dataset[idx][0]
        loss = batch_loss(params, src)
        for trial in range(num_perturb):
            params_perturb = [p + r_perturb * torch.randn_like(p) if names[i] == perturb_name else p for i, p in enumerate(params)]
            loss_perturb = batch_loss(params_perturb, src)
            diff.append(loss_perturb - loss)

    return {0: diff}

def get_robustness_subspace(model, criterion, **kwargs):
    dataset = kwargs['dataset']
    num_perturb = kwargs['num_perturb']
    r_perturb = kwargs['r_perturb']
    config = kwargs['config']
    data_sample_size = kwargs['data_sample_size']
    perturb_name = kwargs['perturb_name']
    indices = np.random.permutation(np.arange(len(dataset)))[:data_sample_size]
    names = list(n for n, _ in model.named_parameters())

    def batch_loss(params, src):
        output = torch.func.functional_call(model, {n: p.detach() for n, p in zip(names, params)}, src)
        vocab_size = output.size(-1)
        loss = criterion(
            output[:, :-1].contiguous().view(-1, vocab_size),
            src[:, 1:].contiguous().view(-1),
        )
        return loss

    diff = dict()

    for idx in indices:
        params = list(model.parameters())
        src = dataset[idx][0]
        loss = batch_loss(params, src)
        for trial in range(num_perturb):
            for i, p in enumerate(params):
                if names[i] == perturb_name:
                    U, s, Vt = torch.linalg.svd(p.detach())
                    subspace_perturbations = perturb_orthogonal_subspace(p.detach(), r_perturb, U, Vt)
            for j in range(len(subspace_perturbations)):
                params_perturb = [subspace_perturbations[j] if names[i] == perturb_name else p for i, p in enumerate(params)]
                loss_perturb = batch_loss(params_perturb, src)
                if j not in diff.keys():
                    diff[j] = [loss_perturb - loss]
                else:
                    diff[j].append(loss_perturb - loss)

    return diff

def perturb_subspace(W, r_perturb):
    U, s, Vt = torch.linalg.svd(W)
    ret = []
    for i in range(len(s)):
        s[i] += r_perturb * torch.randn_like(s[i])
        diagS = torch.zeros((U.shape[0], Vt.shape[0]))
        diagS[range(min(U.shape[0], Vt.shape[0])), range(min(U.shape[0], Vt.shape[0]))] = s
        ret.append(U @ diagS @ Vt)
    return ret

def perturb_orthogonal_subspace(W, r_perturb, U, Vt):
    noise = torch.randn_like(W)
    UVt = torch.permute(torch.tensordot(U, Vt, dims=0), (1,2,0,3))
    proj = torch.sum(UVt * noise[None, None, :], dim=(2,3)) # computing a matrix whose element is <noise, u_iv_j^T>
    diag_proj = torch.zeros_like(proj)
    diag_proj[range(min(proj.shape[0], proj.shape[1])), range(min(proj.shape[0], proj.shape[1]))] = proj[range(min(proj.shape[0], proj.shape[1])), range(min(proj.shape[0], proj.shape[1]))]
    proj = U @ diag_proj @ Vt
    ret1 = W + r_perturb * proj

    noise = torch.randn_like(W)
    proj = torch.sum(UVt * noise[None, None, :], dim=(2,3)) # computing a matrix whose element is <noise, u_iv_j^T>
    diag_proj = torch.zeros_like(proj)
    diag_proj[range(min(proj.shape[0], proj.shape[1])), range(min(proj.shape[0], proj.shape[1]))] = proj[range(min(proj.shape[0], proj.shape[1])), range(min(proj.shape[0], proj.shape[1]))]
    proj = U @ diag_proj @ Vt
    err = noise - proj
    ret2 = W + r_perturb * err
    return (ret1, ret2)

def get_robustness_blk(model, criterion, **kwargs):
    dataset = kwargs['dataset']
    num_perturb = kwargs['num_perturb']
    r_perturb = kwargs['r_perturb']
    config = kwargs['config']

    data_sample_size = kwargs['data_sample_size']
    indices = np.random.permutation(np.arange(len(dataset)))[:data_sample_size]

    names = list(n for n, _ in model.named_parameters())

    def batch_loss(params, src):
        output = torch.func.functional_call(model, {n: p.detach() for n, p in zip(names, params)}, src)
        vocab_size = output.size(-1)
        loss = criterion(
            output[:, :-1].contiguous().view(-1, vocab_size),
            src[:, 1:].contiguous().view(-1),
        )
        return loss

    diff = []
    diff_by_blk = dict()
    keywords = ['embed', 'fc'] + [str(i) for i in range(config.num_layers)]
    for keyword in keywords:
        diff_by_blk[keyword] = []

    for idx in indices:
        params = list(model.parameters())
        src = dataset[idx][0]
        loss = batch_loss(params, src)
        for keyword in keywords:
            for trial in range(num_perturb):
                params_perturb = [p + r_perturb * torch.randn_like(p) if keyword in names[i] else p for i, p in enumerate(params)]
                loss_perturb = batch_loss(params_perturb, src)
                diff_by_blk[keyword].append(loss_perturb - loss)
        for trial in range(num_perturb):
            params_perturb = [p + r_perturb * torch.randn_like(p) for p in params]
            loss_perturb = batch_loss(params_perturb, src)
            diff.append(loss_perturb - loss)
    
    for keyword in keywords:
        diff_by_blk[keyword] = sum(diff_by_blk[keyword]) / len(diff_by_blk[keyword])

    return sum(diff) / len(diff), diff_by_blk

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
        src, lens = data.gen_simple_data(
            vocab,
            max_seq_len,
            sample_size,
            return_lens=True,
            rep_l=rep_l,
            rep_h=rep_h,
        )

        return src.to(device), lens, None, None

    elif regime == "varied repetition":
        src, lens, starts, patterns = data.gen_repetition_data(
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

def get_mask(src, lens, starts=None, ignore_segment=0, ignore_burning=0):
    M = torch.ones_like(src)
    if lens is not None and starts is None:
        M = torch.Tensor(
            utils.mask_get_along_axis(
                src.shape,
                lens,
                ignore_segment=ignore_segment,
                ignore_burning=ignore_burning,
            )
        )
    elif lens is not None and starts is not None:
        M = torch.Tensor(
            utils.mask_get_given_starts(
                src.shape,
                lens,
                starts,
                ignore_segment=ignore_segment,
                ignore_burning=ignore_burning,
            )
        )
    return M

if __name__ == "__main__":
    perturb_orthogonal_subspace(torch.randn(3,3), 1)