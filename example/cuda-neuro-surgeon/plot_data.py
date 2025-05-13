import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set the style
plt.style.use('seaborn')
sns.set_palette("husl")

# Read the CSV file
df = pd.read_csv('solar_wind_2015.csv')

# Convert the 'Date & Time' column to datetime
df['Date & Time'] = pd.to_datetime(df['Date & Time'])

# Filter for January 1st, 2015
start_date = '2015-12-01'
end_date = '2015-12-02'
mask = (df['Date & Time'] >= start_date) & (df['Date & Time'] < end_date)
df_filtered = df.loc[mask]

# Create the figure and axis
plt.figure(figsize=(15, 8))

# Plot Solar data (using the first Solar column)
plt.plot(df_filtered['Date & Time'], df_filtered['Solar [kW]'], 
         label='Solar Power', 
         linewidth=2,
         color='#FFA500',  # Orange color for solar
         marker='o')  # Add markers for each data point

# Plot Wind data
plt.plot(df_filtered['Date & Time'], df_filtered['Wind [kW]'], 
         label='Wind Power', 
         linewidth=2,
         color='#4169E1',  # Royal blue color for wind
         marker='o')  # Add markers for each data point

# Customize the plot
plt.title('Solar and Wind Power Generation (January 1st, 2015)', 
          fontsize=16, 
          pad=20)
plt.xlabel('Time', 
           fontsize=12, 
           labelpad=10)
plt.ylabel('Power (kW)', 
           fontsize=12, 
           labelpad=10)

# Format x-axis to show only hours
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend(fontsize=12)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('solar_wind_power_daily.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show() 