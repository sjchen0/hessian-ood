import matplotlib.pyplot as plt
import torch
import numpy as np

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
            norm = sum([torch.linalg.norm(dF_dW_item).item() for dF_dW_item in dF_dW])
            norms.append(norm)
            trace = torch.trace(d2l_dF2)
            traces.append(trace)
        norm_dF_dW.append(sum(norms) / len(norms))
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
    plt.plot(epochs, norm_dF_dW, '-o', label='$\|dF/dW\|$')
    plt.plot(epochs, tr_d2l_dF2, '-o', label='$tr(d^2l/dF^2)$')
    plt.legend()
    plt.savefig("out/outer_product_hess_decompose.png")

    plt.figure()
    plt.plot(epochs, tr_d2l_dF2, '-o')
    plt.savefig("out/tr_d2l_dF2.png")

if __name__ == "__main__":
    plot_outer_product_hess_decompose()