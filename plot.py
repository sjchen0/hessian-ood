import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib
import os


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

def plot_error_curve(prefix="out/"):
    if type(prefix) == str:
        err_arr = np.load(f"{prefix}error.npy")
        labels = [
            "train loss",
            "train error",
            "ID test loss",
            "ID test error",
            "OOD test loss",
            "OOD test error"
        ]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.plot(err_arr[:,0], label=labels[0], alpha=0.4)
        ax1.plot(err_arr[:,2], label=labels[2])
        ax1.plot(err_arr[:,4], label=labels[4])
        ax1.set_title('Loss')
        ax1.legend()
        ax1.set_xscale('log')

        ax2.plot(err_arr[:,1], label=labels[1], alpha=0.4)
        ax2.plot(err_arr[:,3], label=labels[3])
        ax2.plot(err_arr[:,5], label=labels[5])
        ax2.set_title('Error')
        ax2.legend()
        ax2.set_xscale('log')

        plt.tight_layout()
        plt.savefig(f"{prefix}error_curve.pdf")
    
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        for pref in prefix:
            pref_optim = pref.split("/")[1].split("-")[0]
            hyperparam = ""
            if len(pref.split("/")[1].split("-")) > 1:
                hyperparam = fr"($\rho$={float(pref.split("/")[1][len(pref.split("/")[1].split("-")[0])+1:])})"
            if pref_optim == "sgd":
                pref_optim = "SGD"
                hyperparam = ""
            if pref_optim == "adamW":
                pref_optim = "AdamW"
            if pref_optim == "sam":
                pref_optim = "SAM"
            err_arr = np.load(f"{pref}error.npy")
            steps = np.arange(len(err_arr))[np.arange(len(err_arr)) % 100 == 0]
            err_arr = err_arr[np.arange(len(err_arr)) % 100 == 0]
            labels = [
                "train loss",
                "train error",
                "ID test loss",
                "ID test error",
                "OOD test loss",
                "OOD test error"
            ]
            l1, = ax1.plot(steps, err_arr[:,0], alpha=0.4, linewidth=1)
            # ax1.plot(err_arr[:,2], color=l1.get_color(), linestyle='--')
            ax1.plot(steps, err_arr[:,4], label=f"{pref_optim} {hyperparam}", color=l1.get_color(), linewidth=3, alpha=0.8)

            l2, = ax2.plot(steps, err_arr[:,1], alpha=0.4, linewidth=1)
            # ax2.plot(err_arr[:,3], color=l2.get_color(), linestyle='--')
            ax2.plot(steps, err_arr[:,5], label=f"{pref_optim} {hyperparam}", color=l2.get_color(), linewidth=3, alpha=0.8)

        ax1.set_ylabel('loss', fontsize=20)
        ax1.set_xlabel('epoch', fontsize=20)
        ax1.tick_params(axis='both', labelsize=20)
        ax1.legend(fontsize=16)
        ax1.set_xscale('log')
        ax2.set_ylabel('error', fontsize=20)
        ax2.set_xlabel('epoch', fontsize=20)
        ax2.tick_params(axis='both', labelsize=20)
        ax2.legend(fontsize=16)
        ax2.set_xscale('log')
        plt.tight_layout()
        plt.savefig(f"out/aggregate/error_curve.png", dpi=300)

def plot_sharpness_curve(prefix):
    plt.figure()

    if type(prefix) == str:
        epochs = range(0,20000,500)
        sharpnesses = []
        err_arr = np.load(f"{prefix}error.npy")
        ood_errs = []
        for epoch in epochs:
            if os.path.exists(f"{prefix}sharpness-{epoch}.pt"):
                avg_sharpness = torch.load(f"{prefix}sharpness-{epoch}.pt", map_location='cpu')
                sharpnesses.append(avg_sharpness)
                ood_errs.append(err_arr[epoch, 5])
        plt.scatter(sharpnesses, ood_errs, marker='x')
        plt.xlabel("sharpness")
        plt.ylabel("OOD generalization error")
        plt.savefig(f"{prefix}se_corr.pdf")

    else:
        for pref in prefix:
            pref_optim = pref.split("/")[1].split("-")[0]
            hyperparam = ""
            if len(pref.split("/")[1].split("-")) > 1:
                hyperparam = fr"($\rho$={float(pref.split("/")[1][len(pref.split("/")[1].split("-")[0])+1:])})"
            if pref_optim == "sgd":
                pref_optim = "SGD"
                hyperparam = ""
            if pref_optim == "adamW":
                pref_optim = "AdamW"
            if pref_optim == "sam":
                pref_optim = "SAM"
            epochs = range(0,20000,500)
            sharpnesses = []
            active_epochs = []
            for epoch in epochs:
                if os.path.exists(f"{pref}sharpness-{epoch}.pt"):
                    avg_sharpness = torch.load(f"{pref}sharpness-{epoch}.pt", map_location='cpu')
                    sharpnesses.append(avg_sharpness)
                    active_epochs.append(epoch)
            plt.plot(active_epochs, sharpnesses, label=f"{pref_optim} {hyperparam}")
        plt.xlabel("epoch")
        plt.xscale("log")
        plt.ylabel("sharpness")
        plt.legend()
        plt.savefig(f"out/aggregate/sharpness.png", dpi=300)

def plot_sharpness_error_correlation(prefix):
    fig = plt.figure(figsize=(8,6))

    if type(prefix) == str:
        epochs = range(0,20000,500)
        sharpnesses = []
        err_arr = np.load(f"{prefix}error.npy")
        ood_errs = []
        for epoch in epochs:
            if os.path.exists(f"{prefix}sharpness-{epoch}.pt"):
                avg_sharpness = torch.load(f"{prefix}sharpness-{epoch}.pt", map_location='cpu')
                sharpnesses.append(avg_sharpness)
                ood_errs.append(err_arr[epoch, 5])
        plt.scatter(sharpnesses, ood_errs, marker='x')
        plt.xlabel("sharpness")
        plt.ylabel("OOD generalization error")
        plt.savefig(f"{prefix}se_corr.pdf")

    else:
        for i, pref in enumerate(prefix):
            pref_optim = pref.split("/")[1].split("-")[0]
            hyperparam = ""
            if len(pref.split("/")[1].split("-")) > 1:
                hyperparam = fr"($\rho$={float(pref.split("/")[1][len(pref.split("/")[1].split("-")[0])+1:])})"
            if pref_optim == "sgd":
                pref_optim = "SGD"
                hyperparam = ""
            if pref_optim == "adamW":
                pref_optim = "AdamW"
            if pref_optim == "sam":
                pref_optim = "SAM"
            epochs = range(0,20000,500)
            sharpnesses = []
            err_arr = np.load(f"{pref}error.npy")
            ood_errs = []
            for epoch in epochs:
                if os.path.exists(f"{pref}sharpness-{epoch}.pt"):
                    avg_sharpness = torch.load(f"{pref}sharpness-{epoch}.pt", map_location='cpu')
                    sharpnesses.append(avg_sharpness)
                    ood_errs.append(err_arr[epoch, 5])
            
            alphas = np.linspace(0.2, 1.0, len(sharpnesses))
            colors = [(*plt.get_cmap("tab10")(i)[:3], alpha) for alpha in alphas]
            plt.plot(sharpnesses, ood_errs, alpha=0.2, linewidth=3)
            plt.scatter(sharpnesses, ood_errs, color=colors, label=f"{pref_optim} {hyperparam}", marker='x', s=100, linewidth=3)

        plt.xlabel("sharpness", fontsize=20)
        # plt.xscale("log")
        plt.ylabel("OOD generalization error", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid()
        plt.tight_layout()

        legend = plt.legend(fontsize=16)

        # Get legend handle
        handles = legend.legend_handles

        # Change marker properties
        for i in range(len(handles)):
            handles[i].set_facecolor((*plt.get_cmap("tab10")(i)[:3], 1))

        plt.savefig(f"out/aggregate/se_corr.png", dpi=300)

def plot_rotation_sharpness(prefix):
    if type(prefix) == str:
        # fig, axes = plt.subplots(4, 1, figsize=(12, 8))
        plt.figure(figsize=(8, 6))
        epochs = range(20000)
        active_epochs = []
        sharpness_self = []
        sharpness_rot_total = []
        colors = plt.get_cmap("tab10").colors
        cnt = 0
        for epoch in epochs:
            if os.path.exists(f"{prefix}robustness-{epoch}.pt"):
                sharpness_rot = []
                avg_sharpness = torch.load(f"{prefix}robustness-{epoch}.pt", map_location='cpu')
                avg_sharpness_alter = torch.load(f"{prefix}sharpness-{epoch}.pt", map_location='cpu')
                sharpness_self.append(max(avg_sharpness.item(), avg_sharpness_alter.item()))
                for trial in range(50):
                    if os.path.exists(f"{prefix}robustness-rotate-{epoch}-trial-{trial}.pt"):
                        avg_sharpness = torch.load(f"{prefix}robustness-rotate-{epoch}-trial-{trial}.pt", map_location='cpu')
                        sharpness_rot.append(avg_sharpness)
                sharpness_rot_total.append(sharpness_rot)
                active_epochs.append(epoch)
                plt.scatter([epoch-100]*len(sharpness_rot), sharpness_rot, marker='x', alpha=0.2, linewidth=3, color=colors[1])
                #axes[cnt].hist(sharpness_rot, bins=20, density=True, alpha=0.5)
                #axes[cnt].axvline(x=sharpness_self[-1], linestyle='--', linewidth=3, label="trained", color=colors[0])
                cnt += 1
        # plt.plot(active_epochs, sharpness_self, '-s', label="trained", linewidth=2)
        # bar plot
        plt.bar(active_epochs, sharpness_self, width=100, label="trained", color=colors[0])
        sharpness_rot_total = np.array(sharpness_rot_total)
        sharpness_rot_mean = np.mean(sharpness_rot_total, axis=1)
        sharpness_rot_std = np.std(sharpness_rot_total, axis=1)
        # plot mean curve and one std deviation
        # plt.plot(active_epochs, sharpness_rot_mean, '-s', label=r"$W_{QK}^{(2)}$ rotated", linewidth=2)
        # bar plot
        plt.bar(np.array(active_epochs) + 100, sharpness_rot_mean, width=100, label=r"$W_{QK}^{(2)}$ rotated", color=colors[1])
        # plt.fill_between(active_epochs, sharpness_rot_mean - sharpness_rot_std, sharpness_rot_mean + sharpness_rot_std, alpha=0.2, color=colors[1])
        # plot error bar
        plt.errorbar(np.array(active_epochs) + 100, sharpness_rot_mean, yerr=sharpness_rot_std, fmt='o', color='black', linewidth=1, capsize=5, label="1 std")
        plt.axhline(y=0, color='black', linewidth=1)
        # colors = plt.get_cmap("tab10").colors
        #for i in range(len(sharpness_self)):
        #    plt.axvline(x=sharpness_self[i], linestyle='--', linewidth=3, label=f"{i}", color=colors[i])
        plt.legend(fontsize=16, ncol=3)
        #plt.ylim(-150, 320)
        plt.xlabel("epoch", fontsize=20)
        plt.ylabel("sharpness", fontsize=20)
        plt.xticks(active_epochs, fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(f"{prefix}rotation-sharpness.png", dpi=300)

if __name__ == "__main__":
    prefix = ["out/adamW/", "out/sgd-1e-3/", "out/sam-1e-1/", "out/sam-1e-2/", "out/sam-2e-1/", "out/sam-5e-1/"]
    # prefix = "out/sgd-1e-3/"
    # plot_outer_product_hess_decompose()
    # plot_error_curve(prefix)
    # plot_sharpness_curve(prefix)
    # plot_sharpness_error_correlation(prefix)
    plot_rotation_sharpness("out/adamW/")