import pandas as pd
import matplotlib.pyplot as plt
font = {'size': 15}
plt.rc('font', **font)
# Prepared data
data = {
    "Run": ["Baseline", "Static Proportional", "Fairshare"],
    "Total_time_s": [205.243233, 130.943127, 123.043127],
    "Total_time_s_std": [4.12, 0.42, 1.24],
    "Avg_latency_s": [0.933080, 0.573632, 0.533632],
    "Avg_latency_s_std": [0.08, 0.21, 0.11],
}

df = pd.DataFrame(data)

# ── Plot ──────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 4))

bar_width = 0.35
x = range(len(df))

# total exec time bars (left y‑axis)
ax1.bar([i - bar_width/2 for i in x],
        df["Total_time_s"],
        width=bar_width,
        yerr=df["Total_time_s_std"],
        label="Total execution time (s)")

ax1.set_ylabel("Total execution time (s)")
ax1.set_xticks(x)
ax1.set_xticklabels(df["Run"])

# avg latency bars on secondary y‑axis
ax2 = ax1.twinx()
ax2.bar([i + bar_width/2 for i in x],
        df["Avg_latency_s"],
        width=bar_width,
        yerr=df["Avg_latency_s_std"],
        label="Average latency / launch (s)",
        color="tab:orange")

ax2.set_ylabel("kernel launch latency (ms)")

# Combine legends
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc="upper right")

plt.tight_layout()
plt.savefig("plot_cuda_scheduler.pdf")
