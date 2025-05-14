import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
font = {'size':20}
plt.rc('font', **font)
# Create the bar chart
with open('data.csv', 'r') as f:
    data_str = f.read()
df = pd.read_csv(StringIO(data_str), parse_dates=['DateTime'])

# Plot
fig, ax1 = plt.subplots(figsize=(14, 6))

# Left y‑axis: Solar & Wind
ax1.plot(df['DateTime'], df['Solar_kW'], marker='o', label='Solar Power (kW)', color='tab:orange')
ax1.plot(df['DateTime'], df['Wind_kW'],  marker='o', label='Wind  Power (kW)', color='tab:blue')
ax1.set_xlabel('Time')
ax1.set_ylabel('Power (kW)')
ax1.tick_params(axis='x', rotation=45)

# Right y‑axis: Tokens/s (Tok1, Tok2)
ax2 = ax1.twinx()
ax2.plot(df['DateTime'], df['Tok1'], marker='x', linestyle='--', label='Tok1 (tokens/s)', color='tab:green')
ax2.plot(df['DateTime'], df['Tok2'], marker='x', linestyle='--', label='Tok2 (tokens/s)', color='tab:red')
ax2.set_ylabel('Tokens / second')

# Title & legends
lns1, labs1 = ax1.get_legend_handles_labels()
lns2, labs2 = ax2.get_legend_handles_labels()
ax1.legend(lns1 + lns2, labs1 + labs2, loc='upper left')

fig.tight_layout()
plt.savefig('plot_data.pdf', dpi=300, bbox_inches='tight')