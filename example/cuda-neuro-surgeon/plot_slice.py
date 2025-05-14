import numpy as np
import matplotlib.pyplot as plt
font = {'size': 20}
plt.rc('font', **font)
# Data definitions
workloads_data = {
    'MNIST': {
        'slices': ["gpu", 1, 1, 1, 2, 2, 3, 3, 3, "cpu"],
        'latencies': [7.12, 6.22, 6.22, 6.22, 1.42, 1.42, 1.38, 1.38, 1.38, 1.22]
    },
    'ResNet': {
        'slices': ["gpu", 2, 3, 4, 5, 6, 7, 8, 9, "cpu"],
        'latencies': [32.1, 33.1, 34.2, 41.3, 57.5, 119.6, 124.6, 127.5, 130.5, 133.5]
    },
    'LLaMA': {
        'slices': ["gpu", 5, 10, 15, 20, 25, 30, 35, 40, "cpu"],
        'latencies': [287.83, 361.41, 435.00, 508.58, 582.16, 655.74, 729.32, 802.90, 876.48, 950.06]
    }
}

plt.rcParams.update({'font.size': 16})

# Bigger figure to enlarge squares
fig, axes = plt.subplots(3, 1, figsize=(14, 6), constrained_layout=True)

for ax, (name, data) in zip(axes, workloads_data.items()):
    slices     = data['slices']
    latencies  = np.array(data['latencies'])
    
    # percentage of max latency for colouring
    perc = (latencies / latencies.max()) * 100
    heatmap = perc.reshape(1, -1)
    
    # draw heatmap
    im = ax.imshow(heatmap, aspect='equal', vmin=0, vmax=100, cmap='viridis')
    
    # ticks
    ax.set_xticks(range(len(slices)))
    ax.set_xticklabels(slices, rotation=45, ha='right')
    ax.set_yticks([0])
    ax.set_yticklabels([name])
    
    # annotate with latency numbers
    for j, (v, p) in enumerate(zip(latencies, perc)):
        ax.text(j, 0, f'{v:.2f}', va='center', ha='center', color='white' if p > 50 else 'black', fontsize=14)

# shared colorbar
cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.025, pad=0.02)
cbar.set_label('% of max latency')

axes[-1].set_xlabel('# of slices')

plt.savefig('slice_plot.pdf', dpi=300, bbox_inches='tight')