import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib

def plot_outer_product_hess():
    epochs = list(range(0, 13001, 1000))
    tr_H_out = []
    tr_H = []
    for epoch in epochs:
        H_out, H = torch.load(f"out/out-hess-{epoch}.pt", map_location='cpu')
        tr_H_out.append(sum(map(lambda h: torch.trace(h)/50, H_out)).item())
        tr_H.append(sum(map(lambda h: torch.trace(h), H)).item())
    tr_H_out = np.array(tr_H_out)
    tr_H = np.array(tr_H)

    plt.figure()
    plt.plot(epochs, tr_H_out, '-o', label="$tr(H_{out})$")
    plt.plot(epochs, tr_H, '-o', label='tr(H)')
    # plt.plot(tr_H - tr_H_out, '-o', label='tr(H_{functional})')
    plt.legend()
    plt.savefig("out/outer_product_hess.png")

def plot_outer_product_hess_decompose():
    epochs = list(range(0, 10001, 1000))
    tr_H_out = []
    norm_dF_dW = []
    tr_d2l_dF2 = []
    tr_H = []
    for epoch in epochs:
        H_out_decompose, H = torch.load(f"out/out-hess-decompose-{epoch}.pt", map_location='cpu')
        H_out, dF_dW_total, d2l_dF2_total = H_out_decompose

        norms = []
        traces = []
        for dF_dW, d2l_dF2 in zip(dF_dW_total, d2l_dF2_total):
            norm = [torch.linalg.norm(dF_dW_item).item() for dF_dW_item in dF_dW]
            norms.append(norm)
            trace = torch.trace(d2l_dF2)
            traces.append(trace)
        
        norms = np.array(norms)
        norms = np.mean(norms, axis=0)
        norm_dF_dW.append(norms)
        tr_d2l_dF2.append(sum(traces) / len(traces))

        tr_H_out.append(sum(map(lambda h: torch.trace(h)/50, H_out)).item())
        tr_H.append(sum(map(lambda h: torch.trace(h), H)).item())
    tr_H_out = np.array(tr_H_out)
    tr_H = np.array(tr_H)
    norm_dF_dW = np.array(norm_dF_dW)
    tr_d2l_dF2 = np.array(tr_d2l_dF2)

    fig1, ax1 = plt.subplots()
    ax1.plot(epochs, tr_H_out, linewidth=3, label="$tr(H_{out})$")
    ax1.plot(epochs, tr_H, linewidth=3, label='tr(H)')
    ax1.plot(epochs, np.sum(norm_dF_dW, axis=1), linewidth=3, label='$\|dX^{(L)}/dW\|$')
    ax1.plot(epochs, tr_d2l_dF2, linewidth=3, label='$tr(d^2f/dX^{(L)2})$')
    ax1.set_xscale('log')
    #ax1.set_xticks(epochs)
    #ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xlabel('Training steps')
    ax1.legend()
    plt.savefig("out/outer_product_hess_decompose.png")

    if False:
        plt.figure()
        names = ['embed.embed.weight', 'h.0.mha.W_q.weight', 'h.0.mha.W_q.bias', 'h.0.mha.W_k.weight', 'h.0.mha.W_k.bias', 'h.0.mha.W_v.weight', 'h.0.mha.W_v.bias', 'h.0.mha.W_o.weight', 'h.0.mha.W_o.bias', 'h.1.mha.W_q.weight', 'h.1.mha.W_q.bias', 'h.1.mha.W_k.weight', 'h.1.mha.W_k.bias', 'h.1.mha.W_v.weight', 'h.1.mha.W_v.bias', 'h.1.mha.W_o.weight', 'h.1.mha.W_o.bias', 'fc.weight', 'fc.bias']
        for norms_block, name in zip(norm_dF_dW.T, names):
            if "bias" in name:
                continue
            plt.plot(epochs, norms_block, '-o', label=name)
        plt.legend()
        plt.savefig("out/outer_product_hess_block.png")

def plot_outer_product_hess_random_alignment():
    epochs = list(range(6))
    tr_H_out = []
    norm_dF_dW = []
    tr_d2l_dF2 = []
    tr_H = []
    for epoch in epochs:
        H_out_decompose, H = torch.load(f"out/out-hess-random-{epoch}.pt", map_location='cpu')
        H_out, dF_dW_total, d2l_dF2_total = H_out_decompose

        norms = []
        traces = []
        for dF_dW, d2l_dF2 in zip(dF_dW_total, d2l_dF2_total):
            norm = [torch.linalg.norm(dF_dW_item).item() for dF_dW_item in dF_dW]
            norms.append(norm)
            trace = torch.trace(d2l_dF2)
            traces.append(trace)
        
        norms = np.array(norms)
        norms = np.mean(norms, axis=0)
        norm_dF_dW.append(norms)
        tr_d2l_dF2.append(sum(traces) / len(traces))

        tr_H_out.append(sum(map(lambda h: torch.trace(h)/50, H_out)).item())
        tr_H.append(sum(map(lambda h: torch.trace(h), H)).item())
    tr_H_out = np.array(tr_H_out)
    tr_H = np.array(tr_H)
    norm_dF_dW = np.array(norm_dF_dW)
    tr_d2l_dF2 = np.array(tr_d2l_dF2)

    plt.figure()
    plt.plot(epochs, tr_H_out, '-o', label="$tr(H_{out})$")
    plt.plot(epochs, tr_H, '-o', label='tr(H)')
    plt.plot(epochs, np.sum(norm_dF_dW, axis=1), '-o', label='$\|dF/dW\|$')
    plt.plot(epochs, tr_d2l_dF2, '-o', label='$tr(d^2l/dF^2)$')
    plt.legend()
    plt.savefig("out/outer_product_hess_random.png")

    tr_H_out = []
    norm_dF_dW = []
    tr_d2l_dF2 = []
    tr_H = []
    for epoch in epochs:
        H_out_decompose, H = torch.load(f"out/out-hess-align-{epoch}.pt", map_location='cpu')
        H_out, dF_dW_total, d2l_dF2_total = H_out_decompose

        norms = []
        traces = []
        for dF_dW, d2l_dF2 in zip(dF_dW_total, d2l_dF2_total):
            norm = [torch.linalg.norm(dF_dW_item).item() for dF_dW_item in dF_dW]
            norms.append(norm)
            trace = torch.trace(d2l_dF2)
            traces.append(trace)
        
        norms = np.array(norms)
        norms = np.mean(norms, axis=0)
        norm_dF_dW.append(norms)
        tr_d2l_dF2.append(sum(traces) / len(traces))

        tr_H_out.append(sum(map(lambda h: torch.trace(h)/50, H_out)).item())
        tr_H.append(sum(map(lambda h: torch.trace(h), H)).item())
    tr_H_out = np.array(tr_H_out)
    tr_H = np.array(tr_H)
    norm_dF_dW = np.array(norm_dF_dW)
    tr_d2l_dF2 = np.array(tr_d2l_dF2)

    plt.figure()
    plt.plot(epochs, tr_H_out, '-o', label="$tr(H_{out})$")
    plt.plot(epochs, tr_H, '-o', label='tr(H)')
    plt.plot(epochs, np.sum(norm_dF_dW, axis=1), '-o', label='$\|dF/dW\|$')
    plt.plot(epochs, tr_d2l_dF2, '-o', label='$tr(d^2l/dF^2)$')
    plt.legend()
    plt.savefig("out/outer_product_hess_align.png")

def plot_outer_product_hess_random_alignment_merge():
    epochs = list(range(6))
    tr_H_out = []
    norm_dF_dW = []
    tr_d2l_dF2 = []
    tr_H = []
    for epoch in epochs:
        H_out_decompose, H = torch.load(f"out/out-hess-random-{epoch}.pt", map_location='cpu')
        H_out, dF_dW_total, d2l_dF2_total = H_out_decompose

        norms = []
        traces = []
        for dF_dW, d2l_dF2 in zip(dF_dW_total, d2l_dF2_total):
            norm = [torch.linalg.norm(dF_dW_item).item() for dF_dW_item in dF_dW]
            norms.append(norm)
            trace = torch.trace(d2l_dF2)
            traces.append(trace)
        
        norms = np.array(norms)
        norms = np.mean(norms, axis=0)
        norm_dF_dW.append(norms)
        tr_d2l_dF2.append(sum(traces) / len(traces))

        tr_H_out.append(sum(map(lambda h: torch.trace(h)/50, H_out)).item())
        tr_H.append(sum(map(lambda h: torch.trace(h), H)).item())
    tr_H_out = np.array(tr_H_out)
    tr_H = np.array(tr_H)
    norm_dF_dW = np.array(norm_dF_dW)
    tr_d2l_dF2 = np.array(tr_d2l_dF2)

    plt.figure()
    plt.plot(epochs, np.sum(norm_dF_dW, axis=1), '-o', label='$\|dF/dW\|$ random')
 
    tr_H_out = []
    norm_dF_dW = []
    tr_d2l_dF2 = []
    tr_H = []
    for epoch in epochs:
        H_out_decompose, H = torch.load(f"out/out-hess-align-{epoch}.pt", map_location='cpu')
        H_out, dF_dW_total, d2l_dF2_total = H_out_decompose

        norms = []
        traces = []
        for dF_dW, d2l_dF2 in zip(dF_dW_total, d2l_dF2_total):
            norm = [torch.linalg.norm(dF_dW_item).item() for dF_dW_item in dF_dW]
            norms.append(norm)
            trace = torch.trace(d2l_dF2)
            traces.append(trace)
        
        norms = np.array(norms)
        norms = np.mean(norms, axis=0)
        norm_dF_dW.append(norms)
        tr_d2l_dF2.append(sum(traces) / len(traces))

        tr_H_out.append(sum(map(lambda h: torch.trace(h)/50, H_out)).item())
        tr_H.append(sum(map(lambda h: torch.trace(h), H)).item())
    tr_H_out = np.array(tr_H_out)
    tr_H = np.array(tr_H)
    norm_dF_dW = np.array(norm_dF_dW)
    tr_d2l_dF2 = np.array(tr_d2l_dF2)

   
    plt.plot(epochs, np.sum(norm_dF_dW, axis=1), '-o', label='$\|dF/dW\|$ aligned')

    plt.legend()
    plt.savefig("out/outer_product_hess_merge.png")

if __name__ == "__main__":
    plot_outer_product_hess_decompose()