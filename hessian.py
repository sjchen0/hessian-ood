import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from tqdm import tqdm
from functorch import hessian as f_hessian
from torch.autograd.functional import jacobian
from torch.autograd.functional import hessian

def get_hessian(model, criterion, **kwargs):
    '''
    Compute the full Hessian based on functorch.hessian. May not work on larger models.
    Return: A full Hessian matrix.
    '''
    names = list(n for n, _ in model.named_parameters())
    dataset = kwargs['dataset']
    model_size = sum([p.numel() for p in model.parameters()])
    Hessian = torch.zeros((model_size, model_size))
    data_sample_size = 10
    indices = np.random.permutation(np.arange(len(dataset)))[:data_sample_size]
    indices = [0] # for debug
    data_sample_size = 1 # for debug

    def reduce_shape(M, shape_1, shape_2):
        '''
        Used to flatten a high-dimensional Hessian tensor into a Hessian matrix and ensure symmetry.
        '''
        if len(shape_1) not in (1,2) or len(shape_2) not in (1,2):
            raise NotImplementedError
        if len(shape_1) == 2:
            M = M.flatten(start_dim=0, end_dim=1)
        if len(shape_2) == 2:
            M = M.flatten(start_dim=1, end_dim=2)
        return M

    for idx in indices:
        
        src = dataset[idx][0]

        def batch_loss(params):
            output = torch.func.functional_call(model, {n: p for n, p in zip(names, params)}, src)
            vocab_size = output.size(-1)
            loss = criterion(
                output[:, :-1].contiguous().view(-1, vocab_size),
                src[:, 1:].contiguous().view(-1),
            )
            return loss
        
        hessian_func = f_hessian(batch_loss)
        H = hessian_func(tuple(model.parameters()))
        shapes = [p.shape for p in model.parameters()]
        H = torch.cat([torch.cat([reduce_shape(H[i][j], shapes[i], shapes[j]) for j in range(len(H))], axis=1) for i in range(len(H))], axis=0)
        Hessian += H / data_sample_size

    return Hessian


def get_blkdiag_hessian(model, criterion, **kwargs):
    '''
    Compute a block-diagonal approximation to the Hessian, as full Hessian is large.
    Return: A list of 2x2 tensors, each is the Hessian with respect to a parameter group.
    '''
    dataset = kwargs['dataset']
    data_sample_size = 3
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
        
        # process = psutil.Process()

        for i, (grad, p) in enumerate(zip(grads, params)):
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
                H[i] += dg / data_sample_size
        
    model.zero_grad()
    return H


def get_trace_hessian(model, criterion, **kwargs):
    '''
    Compute the trace of the Hessian based on block-diagonal computations.
    Return: Trace of Hessian evaluated on the loss of randomly drawn data samples.
    '''
    dataset = kwargs['dataset']
    data_sample_size = 5
    indices = np.random.permutation(np.arange(len(dataset)))[:data_sample_size]

    H_traces = []

    for idx in indices:
        src = dataset[idx][0]
        output = model(src)
        vocab_size = output.size(-1)
        loss = criterion(
            output[:, :-1].contiguous().view(-1, vocab_size),
            src[:, 1:].contiguous().view(-1),
        )
        params = tuple(model.parameters())
        grads = autograd.grad(loss, params, create_graph=True)

        H_trace = 0.
        
        # process = psutil.Process()

        for i, (grad, p) in enumerate(zip(grads, params)):
            grad = grad.reshape(-1)
            d = len(grad)
            dg = torch.zeros((d, d))
            for j, g in enumerate(grad):
                g2 = autograd.grad(g, p, retain_graph=True, create_graph=False)[0].view(-1)
                dg[j] = g2
            # print(f"Memory usage: {process.memory_info().vms / 1024**2} MB")
            H_trace += torch.trace(dg)
        
        H_traces.append(H_trace)

    model.zero_grad()
    return H_traces


def get_outer_product_hess(model, criterion, **kwargs):
    '''
    Based on Gauss-Newton decomposition, compute the outer-product hessian
    (\partial F / \partial W) ^ T (\partial^2 l / \partial F^2) (\partial F / \partial W)
    '''
    dataset = kwargs['dataset']
    data_sample_size = 3
    indices = np.random.permutation(np.arange(len(dataset)))[:data_sample_size]

    H_out = None

    for idx_count, idx in enumerate(indices):
        for b in range(len(dataset[0][0])):
            src = dataset[idx][0][b:b+1]
            output = model(src)
            vocab_size = output.size(-1)
            loss = criterion(
                output[:, :-1].contiguous().view(-1, vocab_size),
                src[:, 1:].contiguous().view(-1),
            )
            names = list(n for n, _ in model.named_parameters())
            params = tuple(model.parameters())
            dF_dW = jacobian(
                lambda *par: torch.func.functional_call(model, {n: p for n, p in zip(names, par)}, src),
                params,
                create_graph=False
            )
            dF_dW = [jac.flatten(3,).flatten(0,2) for jac in dF_dW]
            d2l_dF2 = hessian(
                lambda x: criterion(x[:, :-1].contiguous().view(-1, vocab_size), src[:, 1:].contiguous().view(-1)),
                output.detach(),
                create_graph=False
            )
            d2l_dF2 = d2l_dF2.flatten(3,5).flatten(0,2)
            H_out_sample = [jac.T @ d2l_dF2 @ jac for jac in dF_dW]
            if H_out is None:
                H_out = [h / len(indices) / len(src) for h in H_out_sample]
            else:
                H_out = [H + h / len(indices) / len(src) for H, h in zip(H_out, H_out_sample)]
    
    return H_out


def get_outer_product_hess_decompose(model, criterion, **kwargs):
    '''
    Based on Gauss-Newton decomposition, compute the outer-product hessian
    (\partial F / \partial W) ^ T (\partial^2 l / \partial F^2) (\partial F / \partial W)
    '''
    dataset = kwargs['dataset']
    data_sample_size = 3
    indices = np.random.permutation(np.arange(len(dataset)))[:data_sample_size]

    H_out = None
    dF_dW_total = []
    d2l_dF2_total = []

    for idx_count, idx in enumerate(indices):
        for b in range(len(dataset[0][0])):
            src = dataset[idx][0][b:b+1]
            output = model(src)
            vocab_size = output.size(-1)
            loss = criterion(
                output[:, :-1].contiguous().view(-1, vocab_size),
                src[:, 1:].contiguous().view(-1),
            )
            names = list(n for n, _ in model.named_parameters())
            params = tuple(model.parameters())
            dF_dW = jacobian(
                lambda *par: torch.func.functional_call(model, {n: p for n, p in zip(names, par)}, src),
                params,
                create_graph=False
            )
            dF_dW = [jac.flatten(3,).flatten(0,2) for jac in dF_dW]
            d2l_dF2 = hessian(
                lambda x: criterion(x[:, :-1].contiguous().view(-1, vocab_size), src[:, 1:].contiguous().view(-1)),
                output.detach(),
                create_graph=False
            )
            d2l_dF2 = d2l_dF2.flatten(3,5).flatten(0,2)

            dF_dW_total.append(dF_dW)
            d2l_dF2_total.append(d2l_dF2)

            H_out_sample = [jac.T @ d2l_dF2 @ jac for jac in dF_dW]
            if H_out is None:
                H_out = [h / len(indices) / len(src) for h in H_out_sample]
            else:
                H_out = [H + h / len(indices) / len(src) for H, h in zip(H_out, H_out_sample)]
    
    return H_out, dF_dW_total, d2l_dF2_total

def get_inner_rep_norm(model, criterion, **kwargs):
    dataset = kwargs['dataset']
    data_sample_size = 3
    indices = np.random.permutation(np.arange(len(dataset)))[:data_sample_size]
    for idx_count, idx in enumerate(indices):
        src = dataset[idx][0]
        

def get_robustness(model, criterion, **kwargs):
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

    for idx in indices:
        params = list(model.parameters())
        src = dataset[idx][0]
        loss = batch_loss(params, src)
        for trial in range(num_perturb):
            params_perturb = [p + r_perturb * torch.randn_like(p) for p in params]
            loss_perturb = batch_loss(params_perturb, src)
            diff.append(loss_perturb - loss)

    return diff


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