import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# CONFIGURABLE PARAMETER FOR OUR SOLUTION
OUR_SOLUTION_REDUCTION_RATE = 6.04  # Percentage reduction per year

# Color scheme
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#C73E1D',
    'background': '#F5F5F5',
    'text': '#2C3E50',
    'gray': '#9CA3AF'
}

# Set matplotlib parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']

# Load the GeoJSON data
with open('Air Quality Diffusion Tubes/Air Quality Diffusion Tubes.geojson', 'r') as f:
    data = json.load(f)

# Process data to calculate yearly averages
yearly_averages = {}
all_years = list(range(2003, 2025))

for year in all_years:
    year_values = []
    key = f'no2_{year}'

    for feature in data['features']:
        props = feature['properties']
        if props.get('status') != 'Archive':
            value = props.get(key)
            if value is not None and value > 0:
                year_values.append(value)

    if year_values:
        yearly_averages[year] = np.mean(year_values)

# Train model on historical data
historical_years = sorted(yearly_averages.keys())
historical_values = [yearly_averages[year] for year in historical_years]

X_train = np.array(historical_years).reshape(-1, 1)
y_train = np.array(historical_values)

model = LinearRegression()
model.fit(X_train, y_train)

# Calculate annual reduction rate
annual_reduction = abs(model.coef_[0])
r2 = r2_score(y_train, model.predict(X_train))

# Start from 2024 actual value
actual_2024 = yearly_averages.get(2024, 27.4)

# Generate projections using constant annual reduction (current trend)
projection_years = list(range(2025, 2041))
projected_values = []
current_value = actual_2024

for year in projection_years:
    current_value -= annual_reduction
    projected_values.append(current_value)

# Generate projections for OUR SOLUTION (percentage-based reduction)
our_solution_values = []
current_value_solution = actual_2024

for year in projection_years:
    current_value_solution *= (1 - OUR_SOLUTION_REDUCTION_RATE / 100)
    our_solution_values.append(current_value_solution)

# Get historical data for display (2020-2024)
historical_display_years = []
historical_display_values = []

for year in range(2020, 2025):
    if year in yearly_averages:
        historical_display_years.append(year)
        historical_display_values.append(yearly_averages[year])

# Create figure
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('white')

# Plot seamless line from 2020 to 2040
# Combine historical and projected data
all_years_display = historical_display_years + projection_years
all_values_display = historical_display_values + projected_values

# Plot as one continuous line with different styles for past and future
# Historical part (solid)
ax.plot(historical_display_years, historical_display_values,
        color=COLORS['primary'], linewidth=3, linestyle='-', zorder=3)

# Future part (dashed) - Current trend
connection_years = [2024] + projection_years
connection_values = [actual_2024] + projected_values
ax.plot(connection_years, connection_values,
        color=COLORS['primary'], linewidth=3, linestyle='--',
        alpha=1.0, zorder=3, label='Current Trend')

# Our Solution (dotted line in red color)
solution_connection_years = [2024] + projection_years
solution_connection_values = [actual_2024] + our_solution_values
ax.plot(solution_connection_years, solution_connection_values,
        color=COLORS['success'], linewidth=3, linestyle=':',
        alpha=1.0, zorder=4, label='Our Solution')

# Add markers for actual data points
ax.scatter(historical_display_years, historical_display_values,
           color='white', s=80, zorder=5,
           edgecolor=COLORS['primary'], linewidth=2.5)

# Add key milestone annotations for current trend
milestones = {
    2025: projected_values[0],
    2030: projected_values[5],
    2035: projected_values[10],
    2040: projected_values[15]
}

for year, value in milestones.items():
    # Current trend marker
    ax.scatter(year, value, color=COLORS['primary'], s=80,
              zorder=6, edgecolor='white', linewidth=1.5)
    # Annotate all years
    ax.annotate(f'{value:.1f}', xy=(year, value),
               xytext=(0, 12), textcoords='offset points',
               ha='center', fontsize=11, color=COLORS['primary'],
               fontweight='bold')

# Add milestone annotations for our solution
solution_milestones = {
    2025: our_solution_values[0],
    2030: our_solution_values[5],
    2035: our_solution_values[10],
    2040: our_solution_values[15]
}

for year, value in solution_milestones.items():
    # Our solution marker (red)
    ax.scatter(year, value, color=COLORS['success'], s=80,
              zorder=6, edgecolor='white', linewidth=1.5)
    # Annotate all years
    ax.annotate(f'{value:.1f}', xy=(year, value),
               xytext=(0, -15), textcoords='offset points',
               ha='center', fontsize=11, color=COLORS['success'],
               fontweight='bold')

# Add WHO guidelines with better visibility
ax.axhline(y=10, color=COLORS['success'], linestyle='--', alpha=0.5, linewidth=2)

# Add text for WHO lines (right side) with larger font
ax.text(2040.5, 10, 'WHO Target', fontsize=11,
        color=COLORS['success'], ha='left', va='center', fontweight='1000')

# Styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

ax.set_xlabel('Year', fontsize=13, color=COLORS['text'], fontweight='500')
ax.set_ylabel('Air Pollution Level (µg/m³)', fontsize=13, color=COLORS['text'], fontweight='500')
ax.set_title('Air Pollution Trajectory: Sheffield 2020-2040',
            fontsize=16, fontweight='bold', color=COLORS['text'], pad=20)

ax.set_xlim(2019, 2041.5)
ax.set_ylim(0, 45)
ax.set_xticks(range(2020, 2041, 2))

# Subtle grid
ax.grid(True, alpha=0.1, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Add info box with larger text
info_text = f'Current Trend: {annual_reduction:.2f} µg/m³/year\n'
info_text += f'Our Solution: {OUR_SOLUTION_REDUCTION_RATE:.2f}% per year\n'
info_text += f'Model R² = {r2:.3f}'

ax.text(0.02, 0.97, info_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', fontweight='500',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                 edgecolor=COLORS['primary'], linewidth=2, alpha=0.95))

# Add legend with larger font
ax.legend(loc='upper right', fontsize=12, frameon=True,
         facecolor='white', edgecolor='gray', framealpha=0.95,
         title='Projections', title_fontsize=13)

# Add vertical divider at 2024
ax.axvline(x=2024.5, color=COLORS['gray'], linestyle='-',
          alpha=0.2, linewidth=1, zorder=1)
ax.text(2022, 2, 'Historical', fontsize=9, color=COLORS['gray'],
       ha='center', alpha=0.6)
ax.text(2032, 2, 'Projected', fontsize=9, color=COLORS['gray'],
       ha='center', alpha=0.6)

plt.tight_layout()
plt.savefig('results/sheffield_air_pollution_projection_2020_2040.png', dpi=300, bbox_inches='tight',
           facecolor='white')
# plt.show()  # Comment out to prevent hanging

# Print summary
print("\n" + "="*70)
print("AIR POLLUTION REDUCTION FORECAST: SHEFFIELD 2020-2040")
print("="*70)

print(f"\nMODEL PERFORMANCE")
print(f"  R² Score: {r2:.3f}")
print(f"  Annual Reduction: {annual_reduction:.2f} µg/m³/year")
print(f"  Relative Change: {annual_reduction/actual_2024*100:.1f}% per year")

print(f"\nCURRENT TREND PROJECTIONS")
print(f"  2024 (Baseline): {actual_2024:.1f} µg/m³")
print(f"  2025:           {milestones[2025]:.1f} µg/m³ ({(actual_2024-milestones[2025])/actual_2024*100:.0f}% reduction)")
print(f"  2030:           {milestones[2030]:.1f} µg/m³ ({(actual_2024-milestones[2030])/actual_2024*100:.0f}% reduction)")
print(f"  2035:           {milestones[2035]:.1f} µg/m³ ({(actual_2024-milestones[2035])/actual_2024*100:.0f}% reduction)")
print(f"  2040:           {milestones[2040]:.1f} µg/m³ ({(actual_2024-milestones[2040])/actual_2024*100:.0f}% reduction)")

print(f"\nOUR SOLUTION PROJECTIONS ({OUR_SOLUTION_REDUCTION_RATE}% annual reduction)")
print(f"  2025:           {solution_milestones[2025]:.1f} µg/m³ ({(actual_2024-solution_milestones[2025])/actual_2024*100:.0f}% reduction)")
print(f"  2030:           {solution_milestones[2030]:.1f} µg/m³ ({(actual_2024-solution_milestones[2030])/actual_2024*100:.0f}% reduction)")
print(f"  2035:           {solution_milestones[2035]:.1f} µg/m³ ({(actual_2024-solution_milestones[2035])/actual_2024*100:.0f}% reduction)")
print(f"  2040:           {solution_milestones[2040]:.1f} µg/m³ ({(actual_2024-solution_milestones[2040])/actual_2024*100:.0f}% reduction)")

if solution_milestones[2040] < 10:
    print(f"\n✓ Our Solution achieves WHO Target (10 µg/m³) by 2040!")
elif milestones[2040] < 10:
    print(f"\n✓ Current trend achieves WHO Target (10 µg/m³) by 2040")
else:
    years_to_target_current = int((actual_2024 - 10) / annual_reduction) + 2024
    # Calculate years for our solution
    remaining = actual_2024
    year_count = 2024
    while remaining > 10 and year_count < 2100:
        remaining *= (1 - OUR_SOLUTION_REDUCTION_RATE / 100)
        year_count += 1
    print(f"\n→ Current trend reaches WHO Target by {years_to_target_current}")
    print(f"→ Our Solution reaches WHO Target by {year_count}")

print("\n" + "="*70)