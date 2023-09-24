import matplotlib.pyplot as plt

# Categories
categories = ["Syscall Tracepoint", "Uprobe", "Uretprobe"]

# Time values for Kernel and Userspace
kernel_times = [1499.47708, 4751.462610, 5899.706820]
userspace_times = [1489.04251, 445.169770, 472.972220]

bar_width = 0.35
index = range(len(categories))

plt.figure(figsize=(12, 7))

# Plot bars for Kernel and Userspace
bar1 = plt.bar(index, kernel_times, bar_width, color='b', label='Kernel')
bar2 = plt.bar([i + bar_width for i in index], userspace_times, bar_width, color='r', label='Userspace')

# Labeling the figure
plt.xlabel('Probe/Tracepoint Types')
plt.ylabel('Avg Time (ns)')
plt.title('Comparison of Kernel vs. Userspace for Different Probe/Tracepoint Types')
plt.xticks([i + bar_width/2 for i in index], categories)
plt.legend()

plt.tight_layout()
plt.grid(True, which="both", ls="--", c="0.65")
plt.savefig("trace_overhead.png")
plt.show()
