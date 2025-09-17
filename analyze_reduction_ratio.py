#!/usr/bin/env python3
"""
Filter Air Quality Projection by WHO Exceedance with Year-over-Year Improvement Tracking
Creates yearly GeoJSON files containing only points that exceed WHO guidelines
Each file contains full historical data up to that year
"""

import json
import os
from copy import deepcopy
from pathlib import Path
import argparse

WHO_GUIDELINES = {
    'NO2': 10,      # µg/m³
    'PM2.5': 5,     # µg/m³
    'PM10': 15      # µg/m³
}


def load_geojson(filepath: str) -> dict:
    """Load GeoJSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_geojson(data: dict, filepath: str):
    """Save GeoJSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def filter_exceeding_points_by_period(
    geojson_data: dict,
    start_year: int = 2024,
    target_year: int = 2040,
    reduction_ratio: float = 0.03,
    period_gap: int = 5,
    baseline_year: int = 2024
) -> dict:
    """
    Filter GeoJSON to only include points exceeding WHO NO2 guideline for specified period intervals.
    Each period's file contains full history from baseline_year to current period.

    Args:
        geojson_data: Combined projection GeoJSON data
        start_year: Starting year for output files (e.g., 2025)
        target_year: Target year
        reduction_ratio: Annual reduction ratio (e.g., 0.03 = 3%)
        period_gap: Year gap between saved files (e.g., 5 = every 5 years)
        baseline_year: Base year for NO2 values (2024)

    Returns:
        Dictionary with year as key and filtered GeoJSON as value
    """
    yearly_filtered = {}

    # Generate list of years to process (clean intervals from start year)
    years_to_process = []

    # If start_year is 2025, we want 2025, 2030, 2035, 2040
    if start_year == 2025:
        years_to_process = [2025, 2030, 2035, 2040]
    else:
        # For other cases, include baseline year if it's the start year
        if start_year == baseline_year:
            years_to_process.append(baseline_year)

        # Add period intervals
        current = start_year if start_year != baseline_year else start_year + period_gap
        while current <= target_year:
            if current not in years_to_process:
                years_to_process.append(current)
            current += period_gap

        # Ensure target year is included
        if target_year not in years_to_process:
            years_to_process.append(target_year)

    for current_year in years_to_process:
        # Create a new GeoJSON for this year
        year_data = {
            "type": "FeatureCollection",
            "name": f"Diffusion_Tubes_Exceeding_{current_year}_with_History",
            "crs": geojson_data.get("crs"),
            "metadata": {
                "current_year": current_year,
                "baseline_year": baseline_year,
                "annual_reduction_ratio": reduction_ratio,
                "who_guideline_no2": WHO_GUIDELINES['NO2'],
                "years_from_baseline": current_year - baseline_year,
                "cumulative_reduction_factor": (1 - reduction_ratio) ** (current_year - baseline_year),
                "description": f"Points exceeding WHO NO2 guideline in {current_year} with reduction from {baseline_year} baseline"
            },
            "features": []
        }

        # Track statistics for this year
        total_points = 0
        improved_points = 0
        worsened_points = 0

        # Filter features that exceed WHO guideline for this specific year
        for feature in geojson_data['features']:
            props = feature['properties']

            # Get NO2 value for current year
            if current_year == start_year:
                no2_current = props.get(f'no2_{current_year}')
                exceeds_current = props.get('exceeds_who_no2', False)
            else:
                no2_current = props.get(f'no2_{current_year}')
                exceeds_current = props.get(f'exceeds_who_no2_{current_year}', False)

            # Only include if exceeds WHO guideline in current year
            if exceeds_current and no2_current is not None and no2_current > WHO_GUIDELINES['NO2']:
                total_points += 1

                # Create feature with comprehensive historical data
                filtered_feature = {
                    "type": "Feature",
                    "geometry": feature['geometry'],
                    "properties": {
                        # Basic information
                        "objectid": props.get('objectid'),
                        "defrasiteid": props.get('defrasiteid'),
                        "defrasitename": props.get('defrasitename'),
                        "lat": props.get('lat'),
                        "long": props.get('long'),
                        "postcode": props.get('postcode'),
                        "status": props.get('status'),

                        # Current year data
                        "current_year": current_year,
                        f"no2_{current_year}": round(no2_current, 2),
                        "who_guideline_no2": WHO_GUIDELINES['NO2'],
                        "exceeds_who_no2": True,
                        "exceedance_amount": round(no2_current - WHO_GUIDELINES['NO2'], 2),

                        # Historical NO2 values from start_year to current_year
                        "no2_history": {}
                    }
                }

                # Add all historical values from start_year to current_year
                for year in range(start_year, current_year + 1):
                    year_value = props.get(f'no2_{year}')
                    if year_value is not None:
                        filtered_feature['properties']['no2_history'][str(year)] = round(year_value, 2)

                # Calculate year-over-year changes
                if current_year > start_year:
                    prev_value = props.get(f'no2_{current_year - 1}')
                    if prev_value and prev_value > 0:
                        yoy_change = no2_current - prev_value
                        yoy_change_pct = (yoy_change / prev_value) * 100

                        filtered_feature['properties']['yoy_change'] = round(yoy_change, 2)
                        filtered_feature['properties']['yoy_change_pct'] = round(yoy_change_pct, 2)
                        filtered_feature['properties']['improved_from_previous_year'] = yoy_change < 0

                        if yoy_change < 0:
                            improved_points += 1
                        else:
                            worsened_points += 1

                        # Expected reduction based on ratio
                        expected_value = prev_value * (1 - reduction_ratio)
                        filtered_feature['properties']['expected_no2'] = round(expected_value, 2)
                        filtered_feature['properties']['vs_expected'] = round(no2_current - expected_value, 2)
                        filtered_feature['properties']['meets_target_reduction'] = no2_current <= expected_value

                # Calculate cumulative change from baseline year
                baseline_value = props.get(f'no2_{baseline_year}')
                if baseline_value and baseline_value > 0:
                    cumulative_change = no2_current - baseline_value
                    cumulative_change_pct = (cumulative_change / baseline_value) * 100

                    filtered_feature['properties']['cumulative_change_from_baseline'] = round(cumulative_change, 2)
                    filtered_feature['properties']['cumulative_change_pct'] = round(cumulative_change_pct, 2)

                    # Expected cumulative reduction from baseline
                    years_elapsed = current_year - baseline_year
                    expected_cumulative_factor = (1 - reduction_ratio) ** years_elapsed
                    expected_cumulative_value = baseline_value * expected_cumulative_factor
                    filtered_feature['properties']['expected_cumulative_no2'] = round(expected_cumulative_value, 2)
                    filtered_feature['properties']['cumulative_target_met'] = no2_current <= expected_cumulative_value

                # Add original 2020-2024 data for reference
                for hist_year in range(2020, 2025):
                    hist_value = props.get(f'no2_{hist_year}')
                    if hist_value:
                        filtered_feature['properties'][f'original_no2_{hist_year}'] = round(hist_value, 2)

                year_data['features'].append(filtered_feature)

        # Add comprehensive summary statistics
        year_data['summary'] = {
            'year': current_year,
            'total_exceeding_points': len(year_data['features']),
            'who_guideline_no2': WHO_GUIDELINES['NO2'],
            'annual_reduction_ratio': reduction_ratio,
            'annual_reduction_pct': reduction_ratio * 100,
            'years_since_baseline': current_year - baseline_year,
            'cumulative_reduction_factor': (1 - reduction_ratio) ** (current_year - baseline_year),
            'improved_from_previous_year': improved_points,
            'worsened_from_previous_year': worsened_points,
            'unchanged_from_previous_year': total_points - improved_points - worsened_points if current_year > start_year else 0
        }

        if year_data['features']:
            # Calculate average statistics
            avg_exceedance = sum(f['properties']['exceedance_amount'] for f in year_data['features']) / len(year_data['features'])
            year_data['summary']['average_exceedance'] = round(avg_exceedance, 2)

            # NO2 statistics
            no2_values = [f['properties'][f'no2_{current_year}'] for f in year_data['features']]
            year_data['summary']['max_no2'] = round(max(no2_values), 2)
            year_data['summary']['min_no2'] = round(min(no2_values), 2)
            year_data['summary']['avg_no2'] = round(sum(no2_values) / len(no2_values), 2)

            # Year-over-year improvement statistics
            if current_year > start_year:
                improvements = [f['properties'].get('yoy_change_pct', 0) for f in year_data['features']
                              if 'yoy_change_pct' in f['properties']]
                if improvements:
                    year_data['summary']['avg_yoy_improvement_pct'] = round(sum(improvements) / len(improvements), 2)

                # Points meeting reduction target
                meeting_target = sum(1 for f in year_data['features']
                                   if f['properties'].get('meets_target_reduction', False))
                year_data['summary']['points_meeting_reduction_target'] = meeting_target
                year_data['summary']['pct_meeting_reduction_target'] = round((meeting_target / total_points * 100) if total_points > 0 else 0, 2)

        yearly_filtered[current_year] = year_data

    return yearly_filtered


def main():
    parser = argparse.ArgumentParser(
        description='Filter air quality projections to show WHO guideline exceeding points with year-over-year tracking'
    )

    parser.add_argument(
        '--input',
        type=str,
        default='results/Air_Quality_Diffusion_Tubes_Projection_2024_2040.geojson',
        help='Input combined projection GeoJSON file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/progress',
        help='Output directory for yearly filtered files'
    )

    parser.add_argument(
        '--start-year',
        type=int,
        default=2025,
        help='Starting year for output files (default: 2025)'
    )

    parser.add_argument(
        '--baseline-year',
        type=int,
        default=2024,
        help='Baseline year for NO2 values (default: 2024)'
    )

    parser.add_argument(
        '--target-year',
        type=int,
        default=2040,
        help='Target year (default: 2040)'
    )

    parser.add_argument(
        '--reduction-ratio',
        type=float,
        default=0.06,
        help='Annual reduction ratio (default: 0.06 = 6%)'
    )

    parser.add_argument(
        '--period-gap',
        type=int,
        default=5,
        choices=[3, 5],
        help='Year gap for period analysis (3 or 5 years, default: 5)'
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load combined projection data
    print(f"Loading projection data from {args.input}...")
    geojson_data = load_geojson(args.input)
    total_features = len(geojson_data['features'])
    print(f"Loaded {total_features} total monitoring locations")

    # Filter by period with improvement tracking
    print(f"\nFiltering exceeding points for {args.period_gap}-year periods from {args.start_year} to {args.target_year}...")
    print(f"Using {args.baseline_year} as baseline year for NO2 values")
    print(f"Applying annual reduction ratio: {args.reduction_ratio * 100:.1f}%")
    print(f"Cumulative reduction after {args.target_year - args.baseline_year} years: {(1 - (1 - args.reduction_ratio) ** (args.target_year - args.baseline_year)) * 100:.1f}%")

    yearly_filtered = filter_exceeding_points_by_period(
        geojson_data,
        start_year=args.start_year,
        target_year=args.target_year,
        reduction_ratio=args.reduction_ratio,
        period_gap=args.period_gap,
        baseline_year=args.baseline_year
    )

    # Save period files and print detailed progress
    print("\n" + "="*100)
    print(f"{args.period_gap}-YEAR PERIOD PROGRESSION OF WHO NO2 GUIDELINE EXCEEDANCES")
    print("="*100)
    print(f"WHO NO2 Guideline: {WHO_GUIDELINES['NO2']} µg/m³")
    print(f"Total monitoring locations: {total_features}")
    print(f"Annual reduction target: {args.reduction_ratio * 100:.1f}%")
    print(f"Files generated for years: {', '.join(str(y) for y in sorted(yearly_filtered.keys()))}")
    print("\n" + "-"*100)
    print("Year | Exceeding | Cumulative Reduction | From Previous Period | Avg NO2 | Max NO2 | Expected vs Actual")
    print("-"*100)

    prev_count = None
    prev_year = None
    initial_count = None

    # Get initial count from baseline year (2024) data
    # Count features that exceed WHO guideline in baseline year
    baseline_exceeding = 0
    for feature in geojson_data['features']:
        props = feature['properties']
        baseline_no2 = props.get(f'no2_{args.baseline_year}')
        if baseline_no2 and baseline_no2 > WHO_GUIDELINES['NO2']:
            baseline_exceeding += 1
    initial_count = baseline_exceeding

    for year in sorted(yearly_filtered.keys()):
        year_data = yearly_filtered[year]

        # Save file
        output_file = output_path / f'Exceeding_Points_{year}.geojson'
        save_geojson(year_data, str(output_file))

        # Get statistics
        summary = year_data['summary']
        count = summary['total_exceeding_points']

        # Calculate cumulative reduction from baseline
        cumulative_reduction = initial_count - count
        cumulative_pct = (cumulative_reduction / initial_count * 100) if initial_count > 0 else 0

        # Calculate expected reduction based on formula from baseline year
        years_elapsed = year - args.baseline_year
        expected_reduction_factor = (1 - args.reduction_ratio) ** years_elapsed
        expected_count = int(initial_count * expected_reduction_factor)
        vs_expected = count - expected_count

        # Calculate changes from previous period
        if prev_count is not None and prev_year is not None:
            period_change = prev_count - count
            period_years = year - prev_year
            period_pct = (period_change / prev_count * 100) if prev_count > 0 else 0
            period_str = f"{period_change:>3} ({period_pct:>5.1f}% in {period_years}y)"
        else:
            period_str = "      Baseline      "

        # Format statistics for display
        avg_no2 = summary.get('avg_no2', 0)
        max_no2 = summary.get('max_no2', 0)

        # Print row
        print(f"{year} | {count:>9} | {cumulative_reduction:>10} ({cumulative_pct:>5.1f}%) | {period_str} | {avg_no2:>7.1f} | {max_no2:>7.1f} | {vs_expected:+3d} vs {expected_count}")

        prev_count = count
        prev_year = year

    # Print final summary
    print("-"*100)
    total_reduction = initial_count - count if initial_count else 0
    total_pct = (total_reduction / initial_count * 100) if initial_count else 0

    print(f"\n=== FINAL SUMMARY ===")
    print(f"Baseline year: {args.baseline_year}")
    print(f"Analysis period: {args.start_year} to {args.target_year}")
    print(f"Annual reduction ratio: {args.reduction_ratio * 100:.1f}%")
    print(f"Expected cumulative reduction from {args.baseline_year}: {(1 - (1 - args.reduction_ratio) ** (args.target_year - args.baseline_year)) * 100:.1f}%")
    print(f"Baseline exceeding points ({args.baseline_year}): {initial_count}")
    print(f"Final exceeding points ({args.target_year}): {count}")
    print(f"Total points reduced: {total_reduction} ({total_pct:.1f}%)")
    print(f"Final percentage exceeding: {count/total_features*100:.1f}% of all locations")

    # Print period-based analysis with actual vs expected
    print(f"\n=== DETAILED PERIOD ANALYSIS ===")
    period_data = []
    sorted_years = sorted(yearly_filtered.keys())

    for i in range(len(sorted_years) - 1):
        start_year = sorted_years[i]
        end_year = sorted_years[i + 1]

        start_data = yearly_filtered[start_year]['summary']
        end_data = yearly_filtered[end_year]['summary']

        period_reduction = start_data['total_exceeding_points'] - end_data['total_exceeding_points']
        period_pct = (period_reduction / start_data['total_exceeding_points'] * 100) if start_data['total_exceeding_points'] > 0 else 0

        # Calculate expected reduction for this period
        period_years = end_year - start_year
        expected_factor = (1 - args.reduction_ratio) ** period_years
        expected_end = int(start_data['total_exceeding_points'] * expected_factor)
        actual_vs_expected = end_data['total_exceeding_points'] - expected_end

        period_info = {
            'period': f"{start_year}-{end_year}",
            'years': period_years,
            'start_points': start_data['total_exceeding_points'],
            'end_points': end_data['total_exceeding_points'],
            'reduction': period_reduction,
            'reduction_pct': period_pct,
            'expected_end': expected_end,
            'vs_expected': actual_vs_expected,
            'avg_no2_start': start_data.get('avg_no2', 0),
            'avg_no2_end': end_data.get('avg_no2', 0)
        }
        period_data.append(period_info)

        print(f"\nPeriod {start_year}-{end_year} ({period_years} years):")
        print(f"  • Points: {start_data['total_exceeding_points']} → {end_data['total_exceeding_points']} (reduced by {period_reduction}, {period_pct:.1f}%)")
        print(f"  • Expected: {start_data['total_exceeding_points']} → {expected_end} (based on {args.reduction_ratio*100:.1f}% annual reduction)")
        print(f"  • Actual vs Expected: {actual_vs_expected:+d} points")
        print(f"  • Avg NO2: {start_data.get('avg_no2', 0):.1f} → {end_data.get('avg_no2', 0):.1f} µg/m³ ({start_data.get('avg_no2', 0) - end_data.get('avg_no2', 0):.1f} reduction)")

    # Sort periods by reduction percentage
    print(f"\n=== PERIODS RANKED BY REDUCTION ===")
    sorted_periods = sorted(period_data, key=lambda x: x['reduction'], reverse=True)
    for i, period in enumerate(sorted_periods, 1):
        print(f"{i}. Period {period['period']}: {period['reduction']} points reduced ({period['reduction_pct']:.1f}% over {period['years']} years)")

    # Print NO2 concentration trends
    print(f"\n=== NO2 CONCENTRATION TRENDS ===")
    sorted_years = sorted(yearly_filtered.keys())
    first_year_data = yearly_filtered[sorted_years[0]]['summary']
    last_year_data = yearly_filtered[sorted_years[-1]]['summary']

    print(f"Average NO2:")
    print(f"  • Start ({args.start_year}): {first_year_data.get('avg_no2', 0):.1f} µg/m³")
    print(f"  • End ({args.target_year}): {last_year_data.get('avg_no2', 0):.1f} µg/m³")
    print(f"  • Total reduction: {first_year_data.get('avg_no2', 0) - last_year_data.get('avg_no2', 0):.1f} µg/m³")
    print(f"  • Reduction percentage: {((first_year_data.get('avg_no2', 0) - last_year_data.get('avg_no2', 0)) / first_year_data.get('avg_no2', 0) * 100):.1f}%")

    print(f"\nMaximum NO2:")
    print(f"  • Start ({args.start_year}): {first_year_data.get('max_no2', 0):.1f} µg/m³")
    print(f"  • End ({args.target_year}): {last_year_data.get('max_no2', 0):.1f} µg/m³")
    print(f"  • Reduction: {first_year_data.get('max_no2', 0) - last_year_data.get('max_no2', 0):.1f} µg/m³")

    print(f"\nAverage exceedance above WHO guideline:")
    print(f"  • Start ({args.start_year}): {first_year_data.get('average_exceedance', 0):.1f} µg/m³")
    print(f"  • End ({args.target_year}): {last_year_data.get('average_exceedance', 0):.1f} µg/m³")
    print(f"  • Improvement: {first_year_data.get('average_exceedance', 0) - last_year_data.get('average_exceedance', 0):.1f} µg/m³")

    print(f"\n=== FORMULA-BASED PROJECTION ACCURACY ===")
    print(f"Annual reduction ratio: {args.reduction_ratio * 100:.1f}%")
    print(f"Formula: Points(year) = Points({args.baseline_year}) × (1 - {args.reduction_ratio})^(year - {args.baseline_year})")

    for year in sorted(yearly_filtered.keys()):
        years_elapsed = year - args.baseline_year
        expected_factor = (1 - args.reduction_ratio) ** years_elapsed
        expected_count = int(initial_count * expected_factor)
        actual_count = yearly_filtered[year]['summary']['total_exceeding_points']
        accuracy = (1 - abs(actual_count - expected_count) / expected_count) * 100 if expected_count > 0 else 100

        print(f"  Year {year} ({years_elapsed} years from baseline): Expected {expected_count}, Actual {actual_count}, Accuracy {accuracy:.1f}%")

    print(f"\nFiles saved to: {output_path}/")

    # Create a comprehensive summary JSON file
    summary_data = {
        "analysis": "WHO NO2 Guideline Exceedance Progression with Year-over-Year Tracking",
        "configuration": {
            "who_guideline_no2": WHO_GUIDELINES['NO2'],
            "annual_reduction_ratio": args.reduction_ratio,
            "annual_reduction_pct": args.reduction_ratio * 100,
            "start_year": args.start_year,
            "target_year": args.target_year,
            "period_gap": args.period_gap
        },
        "overall_results": {
            "total_locations": total_features,
            "initial_exceeding": initial_count,
            "final_exceeding": count,
            "total_reduction": total_reduction,
            "reduction_percentage": round(total_pct, 2),
            "expected_cumulative_reduction_pct": round((1 - (1 - args.reduction_ratio) ** (args.target_year - args.start_year)) * 100, 2)
        },
        "period_analysis": period_data,
        "top_improvement_periods": sorted_periods[:3] if sorted_periods else [],
        "concentration_trends": {
            "avg_no2_start": first_year_data.get('avg_no2', 0),
            "avg_no2_end": last_year_data.get('avg_no2', 0),
            "avg_no2_reduction": first_year_data.get('avg_no2', 0) - last_year_data.get('avg_no2', 0),
            "max_no2_start": first_year_data.get('max_no2', 0),
            "max_no2_end": last_year_data.get('max_no2', 0),
            "avg_exceedance_start": first_year_data.get('average_exceedance', 0),
            "avg_exceedance_end": last_year_data.get('average_exceedance', 0)
        },
        "projection_accuracy": {
            "periods_analyzed": len(sorted_years),
            "formula": f"Points(year) = Points(2024) × (1 - {args.reduction_ratio})^years",
            "accuracy_by_period": []
        },
        "yearly_progression": []
    }

    for year in sorted(yearly_filtered.keys()):
        year_summary = yearly_filtered[year]['summary'].copy()
        summary_data["yearly_progression"].append(year_summary)

    summary_file = output_path / 'exceedance_summary_with_tracking.json'
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"Comprehensive summary saved to: {summary_file}")


if __name__ == "__main__":
    main()