import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_outer_product_hess():
    epochs = list(range(0, 5001, 1000))
    tr_H_out = []
    tr_H = []
    for epoch in epochs:
        H_out, H = torch.load(f"out/out-hess-{epoch}.pt", map_location='cpu')
        tr_H_out.append(sum(map(lambda h: torch.trace(h)/50, H_out)).item())
        tr_H.append(sum(map(lambda h: torch.trace(h), H)).item())
    tr_H_out = np.array(tr_H_out)
    tr_H = np.array(tr_H)

    plt.figure()
    plt.plot(tr_H_out, '-o', label="$tr(H_{out})$")
    plt.plot(tr_H, '-o', label='tr(H)')
    # plt.plot(tr_H - tr_H_out, '-o', label='tr(H_{functional})')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_outer_product_hess()