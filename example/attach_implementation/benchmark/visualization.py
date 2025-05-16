import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib.ticker as ticker

# Load the benchmark results
with open('example/attach_implementation/benchmark/example_benchmark_results.json', 'r') as f:
    results = json.load(f)

# Extract RPS values and standard deviations
rps_data = results['average_results']
std_data = results['standard_deviation']

# Create user-friendly names for the modules
module_names = {
    'no_module': 'No Module',
    'baseline': 'Baseline C Module',
    'wasm': 'WebAssembly Module',
    'lua': 'Lua Module',
    'bpftime': 'BPFtime Module',
    'rlbox': 'RLBox Module',
    'erim': 'ERIM Module'
}

# Special handling: separate "No Module" from other modules
no_module_rps = rps_data['no_module']['rps']
no_module_std = std_data['no_module']['rps']

# Process other modules
other_modules = {k: v for k, v in rps_data.items() if k != 'no_module'}
# Sort other modules by RPS value in descending order
sorted_modules = sorted(other_modules.items(), key=lambda x: x[1]['rps'], reverse=True)

# Prepare data for plotting
modules = []
rps_values = []
std_values = []

# Add "No Module" at the bottom
modules.append(module_names['no_module'])
rps_values.append(no_module_rps)
std_values.append(no_module_std)


for module, data in sorted_modules:
    modules.append(module_names.get(module, module))
    rps_values.append(data['rps'])
    std_values.append(std_data[module]['rps'])

# Create colors similar to the image (mint green, peach, coral)
colors = ['#a8d5d3', '#f2c091', '#e69a8b', '#a8d5d3', '#f2c091', '#e69a8b', '#a8d5d3']

# Create the figure and axes
plt.figure(figsize=(16, 10))
plt.rcParams.update({'font.size': 20})  # Increase base font size

# Plot horizontal bars with error bars
# bars = plt.barh(modules, rps_values, xerr=std_values, color=colors, 
#                 height=0.6, capsize=8, error_kw={'elinewidth': 2})
bars = plt.barh(modules, rps_values, color=colors, 
                height=0.6, capsize=8, error_kw={'elinewidth': 2})

# Add value labels to the bars with larger font
for i, bar in enumerate(bars):
    # Get the current x axis limits
    x_min, x_max = plt.xlim()
    
    # If the bar is long enough, place the label inside the bar
    if bar.get_width() > x_max * 0.75:
        # Place label inside with right alignment and white color for contrast
        plt.text(bar.get_width() - (x_max * 0.02), bar.get_y() + bar.get_height()/2, 
                 f'{rps_values[i]:.1f}', va='center', ha='right', 
                 fontsize=22, fontweight='bold', color='white')
    else:
        # Place label after the bar but ensure it doesn't extend beyond the plot
        label_pos = min(bar.get_width() + (x_max * 0.02), x_max * 0.95)
        plt.text(label_pos, bar.get_y() + bar.get_height()/2, 
                 f'{rps_values[i]:.1f}', va='center', 
                 fontsize=22, fontweight='bold')

# Set chart labels with larger font
plt.xlabel('Requests per Second (RPS)', fontsize=26, labelpad=15)
plt.ylabel('Module Type', fontsize=26, labelpad=15)

# Format x-axis to show thousands properly and increase tick size
plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

# Add grid for better readability
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Adjust layout and save
plt.tight_layout()
plt.savefig('benchmark_comparison.png', dpi=300)
plt.show() 