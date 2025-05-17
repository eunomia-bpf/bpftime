import matplotlib.pyplot as plt
font = {'size': 20}
plt.rc('font', **font)
# Data
workloads = ['gemm', 'stencil', 'spmv']
nvbit = [38.151, 39.435, 40.647]
cuprobe = [0.41, 0.43, 0.38]
native = [0.06, 0.04, 0.1]
nvbit_std = [0.01, 0.01, 0.01]
cuprobe_std = [0.01, 0.01, 0.01]
native_std = [0.01, 0.01, 0.01]

# Positions and width
x = list(range(len(workloads)))
width = 0.2

fig, ax = plt.subplots(figsize=(15, 6))

bars_cuprobe = ax.bar([i - width for i in x], cuprobe, width, yerr=cuprobe_std, label='cuprobe')
bars_native  = ax.bar(x, native, width, yerr=native_std, label='native')
bars_nvbit   = ax.bar([i + width for i in x], nvbit,  width, yerr=nvbit_std, label='nvbit')

# Axis labels and limits
ax.set_xticks(x)
ax.set_xticklabels(workloads)
ax.set_ylabel('Time (ms)')
ax.set_ylim(0, 1)

# Annotate each bar with its true height, allowing text outside the axis
for bars in [bars_cuprobe, bars_native, bars_nvbit]:
    for bar in bars:
        height = bar.get_height()
        height_text = height
        if height > 1:
            height_text = height
            height = 1
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            height, 
            f'{height_text:g}', 
            ha='center', 
            va='bottom',
        )

# Expand top margin for annotations
fig.subplots_adjust(top=0.8)

ax.legend()
plt.tight_layout()
plt.savefig('memtrace.pdf')
plt.show()
