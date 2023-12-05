import os
import sys
import pickle
import numpy as np
from matplotlib import pyplot as plt

# Plot for w_cos

def plot_nc(nc_dt):
    k_lst = ['w_norm', 'h_norm', 'w_cos', 'h_cos', 'wh_cos', 'nc1_cls']

    
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 9))  
    plt.subplots_adjust(hspace=0.5, wspace=0.4)  

    for k, key in enumerate(k_lst):
        ax = axes[k//2, k % 2]
        cos_matrix = nc_dt[key]  # [K, K]

        avg_val = np.mean(cos_matrix)
        min_val = np.min(cos_matrix)
        max_val = np.max(cos_matrix)
        stats_info = f'Avg: {avg_val:.3f}, Min: {min_val:.3f}, Max: {max_val:.3f}'

        if key in ['w_cos', 'h_cos']:
            im = ax.imshow(cos_matrix, cmap='RdBu', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax)
        else:
            ax.bar(np.arange(len(cos_matrix)), cos_matrix)

        ax.set_title(f'{key} ({stats_info})', fontsize=10)  

    return fig