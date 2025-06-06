import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib.ticker as ticker
import matplotlib as mpl

# Configure matplotlib to embed fonts in PDF
mpl.rcParams.update({
    'pdf.fonttype': 42,  # TrueType fonts
    'pdf.use14corefonts': False,
    'font.family': 'sans-serif'
})

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Load the benchmark results
with open('example/attach_implementation/benchmark/example_benchmark_results.json', 'r') as f:
    results = json.load(f)

# Extract RPS values and standard deviations
rps_data = results['average_results']
std_data = results['standard_deviation']

# Create user-friendly names for the modules
module_names = {
    'no_module': 'Native',
    'baseline': 'Baseline C',
    'wasm': 'WebAssembly',
    'lua': 'Lua',
    'bpftime': 'bpftime',
    'rlbox': 'RLBox',
    'erim': 'ERIM'
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
colors = [ '#a8d5d3', '#a8d5d3', '#e69a8b', '#f2c091', '#f2c091', '#f2c091', '#f2c091']

# Create the figure and axes
plt.figure(figsize=(18, 9))
plt.rcParams.update({'font.size': 20})  # Increase base font size

# Plot horizontal bars with error bars
bars = plt.barh(modules, rps_values, color=colors, 
                height=0.6, capsize=8, error_kw={'elinewidth': 2})

# Calculate the maximum x value to ensure proper scaling
max_value = max(rps_values)
margin = max_value * 0.1  # Add 10% margin for the labels

# Set x-axis limits with extended right margin to fit labels
plt.xlim(0, max_value + margin)

# Add value labels to the bars with larger font
for i, bar in enumerate(bars):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
             f'{int(rps_values[i])}', va='center', ha='left', 
             fontsize=35, color='black')

# Set chart labels with larger font
plt.xlabel('Requests per Second (RPS)', fontsize=35, labelpad=15)

# Format x-axis to show thousands properly and increase tick size
plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

# Add grid for better readability
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Adjust layout and save
plt.tight_layout()
plt.savefig('bpftime-nginx-module.pdf', bbox_inches='tight')  # Use tight bounding box
plt.show() 