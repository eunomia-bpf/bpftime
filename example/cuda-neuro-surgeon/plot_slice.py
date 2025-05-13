import numpy as np
import matplotlib.pyplot as plt

# Data definitions
workloads_data = {
    'MNIST': {
        'slices': [1, 1, 1, 2, 2, 3, 3, 3],
        'latencies': [6.22, 6.22, 6.22, 1.42, 1.42, 1.38, 1.38, 1.38]
    },
    'ResNet': {
        'slices': [2, 3, 4, 5, 6, 7, 8, 9],
        'latencies': [32.1, 33.1, 34.2, 41.3, 57.5, 119.6, 124.6, 127.5]
    },
    'LLaMA': {
        'slices': [5, 10, 15, 20, 25, 30, 35, 40],
        'latencies': [287.83, 361.41, 435.00, 508.58, 582.16, 655.74, 729.32, 802.90]
    }
}

fig, axes = plt.subplots(3, 1, figsize=(8, 8), constrained_layout=True)

for ax, (name, data) in zip(axes, workloads_data.items()):
    slices = data['slices']
    lat = np.array(data['latencies'])
    perc = (lat / lat.max()) * 100
    heatmap = perc.reshape(1, -1)
    
    im = ax.imshow(heatmap, aspect='equal', vmin=0, vmax=100)
    ax.set_xticks(range(len(slices)))
    ax.set_xticklabels(slices, rotation=45)
    ax.set_yticks([0])
    ax.set_yticklabels([name])
    ax.set_title(name, pad=10)

# Shared colorbar
cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label('% of max latency')

# Common labels
axes[-1].set_xlabel('# Of slices')

plt.show()