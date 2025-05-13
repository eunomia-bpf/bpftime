import matplotlib.pyplot as plt

# Data
workloads = ['gemm', 'stencil', 'spmv']
nvbit = [38.151, 39.435, 40.647]
cuprobe = [0.40, 0.40, 0.40]
native = [0.06, 0.04, 0.1]

# Positions and width
x = list(range(len(workloads)))
width = 0.2

fig, ax = plt.subplots()

bars_cuprobe = ax.bar([i - width for i in x], cuprobe, width, label='cuprobe')
bars_native  = ax.bar(x, native, width, label='native')
bars_nvbit   = ax.bar([i + width for i in x], nvbit,  width, label='nvbit')

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
