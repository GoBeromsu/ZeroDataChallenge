# Net Zero Data Challenge 2025

## Dynamic Clean Air Zone (CAZ) Management System

### Competition Details
- **Organizers**: YHODA (Yorkshire & Humber Office for Data Analytics), YPIP, University of Sheffield, UKRI ESRC
- **Date**: September 15-19, 2025
- **Venue**: The Diamond, University of Sheffield
- **Sponsored by**: South Yorkshire Sustainability Centre

### Project Overview
This system provides dynamic Clean Air Zone (CAZ) management tools for Sheffield, enabling real-time analysis and projection of air quality improvements. The platform allows policymakers to dynamically adjust CAZ boundaries and parameters based on air quality monitoring data, helping achieve WHO air quality guidelines through data-driven decision making.

## Key Features
- **Dynamic CAZ Boundary Adjustment**: Modify Clean Air Zone boundaries based on real-time air quality data
- **Interactive Dashboard**: Visualize current air quality status and projected improvements
- **Data-Driven Projections**: Simulate various reduction scenarios to optimize CAZ effectiveness
- **Clustering Analysis**: Identify pollution hotspots using K-means clustering for targeted interventions

## Key Findings
- 209 out of 217 monitoring locations exceed WHO NO2 guidelines (10 µg/m³)
- With dynamic CAZ implementation and 6% annual reduction, exceeding locations decrease to 100 by 2040
- Average NO2 concentration reduces from 26.11 to 13.35 µg/m³

## Components

### 1. CAZ Dashboard (Interactive Web Interface)
```bash
streamlit run caz_dashboard.py
```
Provides interactive visualization and analysis of Clean Air Zone data with real-time updates.

### 2. Dynamic CAZ Projection Tool
```bash
python dynamic_caz_projection.py
```
Projects air quality improvements based on dynamic CAZ policies.

#### Custom Parameters
```bash
python dynamic_caz_projection.py \
  --start-year 2025 \
  --baseline-year 2024 \
  --target-year 2040 \
  --reduction-ratio 0.06 \
  --period-gap 5
```

### 3. CAZ Baseline Filter
```bash
python caz_baseline_filter.py
```
Filters and prepares baseline air quality data for CAZ analysis.

### 4. CAZ Map Generator
```bash
python caz_map_generator.py
```
Generates static and interactive maps showing CAZ boundaries and air quality data.

## Parameters

#### Required Input
- `--input`: Input GeoJSON file with projection data (default: `results/Air_Quality_Diffusion_Tubes_Projection_2024_2040.geojson`)

#### Time Configuration
- `--baseline-year`: Baseline year for NO2 calculations (default: 2024)
- `--start-year`: Starting year for output files (default: 2025)
- `--target-year`: Target year (default: 2040)
- `--period-gap`: File generation interval [3, 5] years (default: 5)

#### Reduction Settings
- `--reduction-ratio`: Annual reduction ratio (default: 0.06 = 6%)

## Mathematical Formula

#### Annual Reduction Formula
Assumes pollution levels decrease by a constant ratio each year:

```
NO2(year) = NO2(baseline) × (1 - reduction_ratio)^(year - baseline_year)
```

#### Example Calculation (6% Annual Reduction)
For baseline year 2024 with NO2 concentration of 50 µg/m³:

- Year 2025: 50 × (1 - 0.06)^1 = 50 × 0.94 = 47.00 µg/m³
- Year 2030: 50 × (1 - 0.06)^6 = 50 × 0.6899 = 34.50 µg/m³
- Year 2035: 50 × (1 - 0.06)^11 = 50 × 0.5063 = 25.31 µg/m³
- Year 2040: 50 × (1 - 0.06)^16 = 50 × 0.3716 = 18.58 µg/m³

## WHO Guidelines

#### Air Quality Standards
- **NO2**: 10 µg/m³ (annual mean)
- **PM2.5**: 5 µg/m³ (annual mean)
- **PM10**: 15 µg/m³ (annual mean)

## Output Files

#### GeoJSON Files
Generated files (5-year intervals):
- `results/progress/Exceeding_Points_2025.geojson`
- `results/progress/Exceeding_Points_2030.geojson`
- `results/progress/Exceeding_Points_2035.geojson`
- `results/progress/Exceeding_Points_2040.geojson`

#### Summary File
- `results/progress/exceedance_summary_with_tracking.json`: Comprehensive analysis results

## Analysis Features

#### Period Analysis
Provides the following information for each period:
- Changes in number of exceeding points
- Actual reduction rate vs expected reduction rate
- Average NO2 concentration changes
- Target achievement rate

#### Year-over-Year Tracking
Tracks for each location:
- Improvement/deterioration compared to previous year
- Actual vs expected value comparison
- Cumulative reduction rate calculation
- Exceedance amount above WHO guideline

#### Statistical Summary
- Total monitoring locations: 217
- Initial exceeding points: 209 (baseline 2024)
- Final exceeding points: 100 (projected 2040)
- Total reduction: 109 points (52.15%)

## Example Commands

#### Launch Interactive CAZ Dashboard
```bash
streamlit run caz_dashboard.py
```

#### Run Dynamic CAZ Projection (6% Reduction Scenario)
```bash
python dynamic_caz_projection.py \
  --start-year 2025 \
  --baseline-year 2024 \
  --reduction-ratio 0.06 \
  --period-gap 5
```

This command will:
1. Calculate NO2 values based on 2024 baseline
2. Generate files for years 2025, 2030, 2035, 2040
3. Apply 6% annual reduction
4. Filter only points exceeding WHO guideline (10 µg/m³)

## Expected Results (6% Annual Reduction)

#### Cumulative Reduction
- Year 2025: 209 points (94% of baseline retained)
- Year 2030: 199 points (69% of baseline retained)
- Year 2035: 164 points (51% of baseline retained)
- Year 2040: 100 points (37% of baseline retained)

#### Success Metrics
- Total reduction of 52.15% achieved over 16 years
- Average NO2: 26.11 → 13.35 µg/m³
- Maximum NO2: 56.32 → 22.26 µg/m³

## Data Source
- Sheffield Air Quality monitoring data (2020-2024)
- 217 monitoring locations (diffusion tubes and real-time monitors)
- GeoJSON format with annual NO2 measurements

## Technologies
- Python 3.x
- Streamlit for interactive dashboard
- Folium for dynamic map visualization
- Plotly for interactive charts
- Pandas for data analysis
- Scikit-learn for K-means clustering
- Output: GeoJSON files compatible with GIS tools

## Dynamic CAZ Management Approach
The system enables dynamic Clean Air Zone management through:
1. **Real-time Monitoring**: Continuous analysis of air quality data from 217 monitoring locations
2. **Adaptive Boundaries**: CAZ boundaries that can be adjusted based on pollution patterns
3. **Scenario Modeling**: Test different reduction targets and timelines
4. **Impact Assessment**: Evaluate CAZ effectiveness through data visualization
5. **Policy Optimization**: Data-driven recommendations for CAZ modifications