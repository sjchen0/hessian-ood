import numpy as np
import matplotlib.pyplot as plt
trial_diff = np.load('out/trial_diff_300.npy')
plt.figure()
mean = np.mean(trial_diff, axis=1)
std = np.std(trial_diff, axis=1)
plt.plot(mean * 1e6, linewidth=5)
plt.xscale('log')
#plt.fill_between(np.arange(len(mean)), mean-std, mean+std, alpha=0.1)
plt.axvline(2200, linewidth=3, color='red', linestyle='--', dashes=(3.2,1.5))
plt.title('Loss instability', fontweight='bold', fontsize=24)
plt.savefig('out/loss_instability_2.pdf')