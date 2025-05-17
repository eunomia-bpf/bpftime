import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Host side boxes
host_y = [0.9, 0.75, 0.6, 0.45, 0.3]
host_labels = ["Application", "VFS", "Block Layer", "NVMe Driver", "NIC"]

for y, label in zip(host_y, host_labels):
    rect = patches.Rectangle((0.05, y-0.05), 0.25, 0.08, fill=False)
    ax.add_patch(rect)
    ax.text(0.175, y, label, ha='center', va='center')

# Target side boxes
tgt_y = [0.9, 0.75, 0.6]
tgt_labels = ["NIC", "NVMe Driver", "Block Layer"]

for y, label in zip(tgt_y, tgt_labels):
    rect = patches.Rectangle((0.7, y-0.05), 0.25, 0.08, fill=False)
    ax.add_patch(rect)
    ax.text(0.825, y, label, ha='center', va='center')

# NVMe SSD
rect_ssd = patches.Rectangle((0.75, 0.35), 0.15, 0.08, fill=False)
ax.add_patch(rect_ssd)
ax.text(0.825, 0.39, "NVMe SSD", ha='center', va='center')

# Network cloud
ax.text(0.5, 0.35, "TCP/RDMA\nNetwork", ha='center', va='center', bbox=dict(boxstyle="round", fill=False))

# Arrows host
arrow_props = dict(arrowstyle="->")
for i in range(len(host_y)-1):
    ax.annotate("",
                xy=(0.175, host_y[i+1]+0.01),
                xytext=(0.175, host_y[i]-0.01),
                arrowprops=arrow_props)
# Arrow host to network
ax.annotate("", xy=(0.3, 0.3), xytext=(0.5-0.02, 0.38), arrowprops=arrow_props)

# Arrows network to target
ax.annotate("", xy=(0.7, 0.38), xytext=(0.5+0.02, 0.38), arrowprops=arrow_props)
# Arrows target
for i in range(len(tgt_y)-1):
    ax.annotate("",
                xy=(0.825, tgt_y[i+1]+0.01),
                xytext=(0.825, tgt_y[i]-0.01),
                arrowprops=arrow_props)
# Arrow into SSD
ax.annotate("", xy=(0.825, 0.35), xytext=(0.825, 0.42), arrowprops=arrow_props)

# Overhead labels
overhead_labels = [
    ((0.175, 0.825), "Syscall overhead"),
    ((0.175, 0.675), "VFS lookup overhead"),
    ((0.175, 0.525), "I/O scheduling & copy"),
    ((0.175, 0.375), "Packet encapsulation"),
    ((0.5, 0.41), "Network latency"),
    ((0.7, 0.41), "Packet steering cost"),
    ((0.825, 0.675), "IRQ & DMA overhead"),
    ((0.825, 0.525), "Dispatch overhead"),
    ((0.825, 0.425), "DMA setup\n& transfer"),
]

for (x, y), text in overhead_labels:
    ax.text(x + 0.03, y, text, va='center')

plt.tight_layout()
plt.savefig('io_path.pdf', bbox_inches='tight')