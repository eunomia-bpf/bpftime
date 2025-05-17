import matplotlib.pyplot as plt
font = {'size': 20}
plt.rc('font', **font)
# Data
policies = ['Native', 'LRU Policy', 'LFU Policy']
runtimes = [451.35, 415.29, 426.41]
runtimes_std = [10.29, 16.52, 9.14]

# Create bar chart with wider figure size
fig, ax = plt.subplots(figsize=(10, 6))  # Increase width from default to 10 inches
ax.bar(policies, runtimes, yerr=runtimes_std, capsize=5, width=0.6)  # Make bars wider

# Labels and title
ax.set_ylabel('End-to-End Runtime (seconds)')

# Annotate bars with their values
for i, v in enumerate(runtimes):
    ax.text(i, v + 2, f"{v:.2f}", ha='center')
plt.tight_layout()
plt.savefig('lru_lfu.pdf', bbox_inches='tight')