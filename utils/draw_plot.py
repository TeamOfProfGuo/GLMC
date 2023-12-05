import os
import sys
import pickle
import numpy as np
from matplotlib import pyplot as plt

# Plot for w_cos

def plot_nc(nc_dt):
    k_lst = ['w_norm', 'h_norm', 'w_cos', 'h_cos', 'wh_cos', 'nc1_cls']

    fig, axes = plt.subplots(nrows=3, ncols=2)
    k = 0
    for key in k_lst:
        cos_matrix = nc_dt[key]  # [K, K]

        
        avg_val = np.mean(cos_matrix)
        min_val = np.min(cos_matrix)
        max_val = np.max(cos_matrix)
        stats_info = f'Avg: {avg_val:.3f}, Min: {min_val:.3f}, Max: {max_val:.3f}'

        if key in ['w_cos', 'h_cos']:
            im = axes[int(k//2), int(k %2)].imshow(cos_matrix, cmap='RdBu')
            plt.colorbar(im, ax=axes[int(k//2), int(k %2)])
            im.set_clim(vmin=-1, vmax=1)
        else:
            axes[int(k//2), int(k %2)].bar(np.arange(len(cos_matrix)), cos_matrix)

       
        axes[int(k//2), int(k %2)].set_title(f'{key} ({stats_info})')

        k += 1

    return fig
