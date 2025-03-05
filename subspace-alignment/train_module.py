import torch
import numpy as np
import data
import utils
from torch import autograd
from tqdm import tqdm

@torch.no_grad()
def get_loss(model, criterion, src):
    output = model(src)
    vocab_size = output.size(-1)
    loss = criterion(
        output[:, :-1].contiguous().view(-1, vocab_size),
        src[:, 1:].contiguous().view(-1),
    )
    return loss

@torch.no_grad()
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
        # print(src.device)
        # print([p.device for p in params])
        loss = batch_loss(params, src)
        for trial in range(num_perturb):
            params_perturb = [p + r_perturb * torch.randn_like(p) if names[i] in perturb_name else p for i, p in enumerate(params)]
            # print([p.device for p in params_perturb])
            loss_perturb = batch_loss(params_perturb, src)
            diff.append((loss_perturb - loss).cpu())

    return {0: diff}

def get_blkdiag_hessian(model, criterion, **kwargs):
    '''
    Compute a block-diagonal approximation to the Hessian, as full Hessian is large.
    Return: A list of 2x2 tensors, each is the Hessian with respect to a parameter group.
    '''
    dataset = kwargs['dataset']
    data_sample_size = kwargs['data_sample_size']
    indices = np.random.permutation(np.arange(len(dataset)))[:data_sample_size]

    H = []

    for idx_count, idx in enumerate(indices):
        src = dataset[idx][0]
        output = model(src)
        vocab_size = output.size(-1)
        loss = criterion(
            output[:, :-1].contiguous().view(-1, vocab_size),
            src[:, 1:].contiguous().view(-1),
        )
        params = tuple(model.parameters())
        grads = autograd.grad(loss, params, create_graph=True)
        names = [n for n, _ in model.named_parameters()]
        # process = psutil.Process()

        for i, (grad, p) in enumerate(zip(grads, params)):
            if names[i] == "h.1.mha.W_k.weight":
                grad = grad.reshape(-1)
                d = len(grad)
                dg = torch.zeros((d, d))
                for j, g in enumerate(grad):
                    g2 = autograd.grad(g, p, retain_graph=True, create_graph=False)[0].view(-1)
                    dg[j] = g2
                # print(f"Memory usage: {process.memory_info().vms / 1024**2} MB")
                if idx_count == 0:
                    H.append(dg / data_sample_size)
                else:
                    H[0] += dg / data_sample_size
        
    model.zero_grad()
    return H

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
                if names[i] == "embed.embed.weight":
                    Ue, se, Vte = torch.linalg.svd(p.detach())
                if names[i] in perturb_name:
                    U, s, Vt = torch.linalg.svd(p.detach())
                    subspace_perturbations = perturb_orthogonal_subspace(p.detach(), r_perturb, U, Vt)
            for j in range(len(subspace_perturbations)):
                params_perturb = [subspace_perturbations[j] if names[i] in perturb_name else p for i, p in enumerate(params)]
                loss_perturb = batch_loss(params_perturb, src)
                if j not in diff.keys():
                    diff[j] = [(loss_perturb - loss).cpu()]
                else:
                    diff[j].append((loss_perturb - loss).cpu())

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
    rets = []
    noise = torch.randn_like(W)
    UVt = torch.permute(torch.tensordot(U, Vt, dims=0), (1,2,0,3))
    proj = torch.sum(UVt * noise[None, None, :], dim=(2,3)) # computing a matrix whose element is <noise, u_iv_j^T>
    diag_proj = torch.zeros_like(proj)
    diag_proj[range(min(proj.shape[0], proj.shape[1])), range(min(proj.shape[0], proj.shape[1]))] = proj[range(min(proj.shape[0], proj.shape[1])), range(min(proj.shape[0], proj.shape[1]))]
    proj = U @ diag_proj @ Vt
    rets.append(W + r_perturb * proj)

    '''
    for i in range(U.shape[1]):
        order_0 = [(j + i) % U.shape[1] for j in range(U.shape[1])]
        order_1 = range(U.shape[1])
        noise = torch.randn_like(W)
        proj = torch.sum(UVt * noise[None, None, :], dim=(2,3)) # computing a matrix whose element is <noise, u_iv_j^T>
        diag_proj = torch.zeros_like(proj)
        diag_proj[order_0, order_1] = proj[order_0, order_1]
        proj = U @ diag_proj @ Vt
        rets.append(W + r_perturb * proj)
    '''

    noise = torch.randn_like(W)
    # UVt = torch.permute(torch.tensordot(U, Vt, dims=0), (1,2,0,3))
    proj = torch.sum(UVt * noise[None, None, :], dim=(2,3)) # computing a matrix whose element is <noise, u_iv_j^T>
    diag_proj = torch.zeros_like(proj)
    diag_proj[range(min(proj.shape[0], proj.shape[1])), range(min(proj.shape[0], proj.shape[1]))] = proj[range(min(proj.shape[0], proj.shape[1])), range(min(proj.shape[0], proj.shape[1]))]
    proj = U @ diag_proj @ Vt
    err = noise - proj
    rets.append(W + r_perturb * err)

    return rets

def perturb_coordinate(W, r_perturb):
    rets = []
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            ret = W.clone()
            ret[i,j] += r_perturb * torch.randn_like(ret[i,j])
            rets.append(ret)
    return rets

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