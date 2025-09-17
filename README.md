# Air Quality Projection Analysis Tool

## Net Zero Data Challenge 2025
This project is participating in the **Net Zero Data Challenge 2025**.

- **Organizers**: YHODA (Yorkshire & Humber Office for Data Analytics), YPIP, University of Sheffield, UKRI ESRC
- **Date**: September 15-19, 2025
- **Venue**: The Diamond, University of Sheffield
- **Sponsored by**: South Yorkshire Sustainability Centre

## Overview
This project analyzes Sheffield's air quality monitoring data to project how locations exceeding WHO guidelines will decrease over time. It was developed for the Net Zero Data Challenge 2025.

## Key Findings
- 209 out of 217 monitoring locations exceed WHO NO2 guidelines (10 µg/m³)
- With 6% annual reduction, exceeding locations decrease to 100 by 2040
- Average NO2 concentration reduces from 26.11 to 13.35 µg/m³

## Usage

#### Basic Usage
```bash
python analyze_reduction_ratio.py
```

#### Custom Parameters
```bash
python analyze_reduction_ratio.py \
  --start-year 2025 \
  --baseline-year 2024 \
  --target-year 2040 \
  --reduction-ratio 0.06 \
  --period-gap 5
```

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

## Example Command

#### Standard 6% Reduction Scenario
```bash
python analyze_reduction_ratio.py \
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
- Standard library only (json, argparse, pathlib)
- Output: GeoJSON files compatible with GIS tools