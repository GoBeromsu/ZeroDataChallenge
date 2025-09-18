import json
import os
import sys
from typing import Dict, List, Tuple, Any, Optional

WHO_GUIDELINES = {
    'NO2': 10,      # µg/m³
    'PM2.5': 5,     # µg/m³
    'PM10': 15      # µg/m³
}

def calculate_average(values: List[float]) -> Optional[float]:
    """Calculate average from a list of values, returns None if empty list."""
    if not values:
        return None
    return round(sum(values) / len(values), 2)

def extract_pollutant_values(props: Dict, pollutant_key_pattern: str, years: List[str]) -> List[float]:
    """Extract valid pollutant values for specified years."""
    values = []
    for year in years:
        key = f'{pollutant_key_pattern}_{year}'
        value = props.get(key)
        # Check for valid numeric values (exclude None, 0, and negative values)
        if value is not None and isinstance(value, (int, float)) and value > 0:
            values.append(float(value))
    return values

def process_diffusion_tubes(input_path: str, output_path: str) -> Tuple[int, int, Dict[str, Any]]:
    """
    Process Diffusion Tubes data - ONLY HAS NO2 DATA
    Returns: (filtered_count, total_count, statistics)
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {input_path}")
        return 0, 0, {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}")
        return 0, 0, {}
    except Exception as e:
        print(f"Error reading file: {e}")
        return 0, 0, {}

    filtered_features = []
    all_features_with_avg = []
    years = ['2020', '2021', '2022', '2023', '2024']

    # Statistics tracking
    stats = {
        'total_locations': 0,
        'locations_with_data': 0,
        'locations_exceeding': 0,
        'no2_min': float('inf'),
        'no2_max': 0,
        'no2_values': []
    }

    for feature in data.get('features', []):
        # Validate feature structure
        if not feature.get('geometry') or not feature.get('properties'):
            continue

        stats['total_locations'] += 1
        props = feature['properties']

        # Extract NO2 values
        no2_values = extract_pollutant_values(props, 'no2', years)

        if no2_values:
            stats['locations_with_data'] += 1
            no2_avg = calculate_average(no2_values)

            # Add calculated values to properties
            feature['properties']['avg_no2_2020_2024'] = no2_avg
            feature['properties']['years_with_data'] = len(no2_values)
            feature['properties']['who_guideline_no2'] = WHO_GUIDELINES['NO2']

            # Update statistics
            stats['no2_values'].append(no2_avg)
            stats['no2_min'] = min(stats['no2_min'], no2_avg)
            stats['no2_max'] = max(stats['no2_max'], no2_avg)

            # Check if exceeds WHO guideline
            if no2_avg > WHO_GUIDELINES['NO2']:
                feature['properties']['exceeds_who_no2'] = True
                filtered_features.append(feature)
                stats['locations_exceeding'] += 1
            else:
                feature['properties']['exceeds_who_no2'] = False

            all_features_with_avg.append(feature)

    # Calculate overall average
    if stats['no2_values']:
        stats['no2_overall_avg'] = calculate_average(stats['no2_values'])
    else:
        stats['no2_overall_avg'] = None

    # Prepare output data
    filtered_data = {
        'type': 'FeatureCollection',
        'name': 'Diffusion_Tubes_WHO_Exceeding',
        'crs': data.get('crs'),
        'features': filtered_features,
        'metadata': {
            'source_file': os.path.basename(input_path),
            'years_analyzed': years,
            'who_guideline_no2': WHO_GUIDELINES['NO2'],
            'statistics': stats
        }
    }

    # Save filtered data
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving file: {e}")
        return 0, len(data.get('features', [])), stats

    print(f"Diffusion Tubes: {stats['locations_exceeding']}/{stats['locations_with_data']} locations with data exceed WHO NO2 guideline")
    print(f"  - Total locations: {stats['total_locations']}")
    print(f"  - Locations with valid data: {stats['locations_with_data']}")
    print(f"  - NO2 range: {stats['no2_min']:.1f} - {stats['no2_max']:.1f} µg/m³")

    return stats['locations_exceeding'], stats['total_locations'], stats

def process_realtime_monitors(input_path: str, output_path: str) -> Tuple[int, int, Dict[str, Any]]:
    """
    Process Real-Time Monitors data - HAS NO2, PM2.5, AND PM10 DATA
    Returns: (filtered_count, total_count, statistics)
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {input_path}")
        return 0, 0, {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}")
        return 0, 0, {}
    except Exception as e:
        print(f"Error reading file: {e}")
        return 0, 0, {}

    filtered_features = []
    years = ['2020', '2021', '2022', '2023', '2024']

    # Statistics tracking
    stats = {
        'total_locations': 0,
        'locations_with_data': 0,
        'locations_exceeding': 0,
        'exceeding_no2': 0,
        'exceeding_pm25': 0,
        'exceeding_pm10': 0,
        'exceeding_multiple': 0
    }

    for feature in data.get('features', []):
        # Validate feature structure
        if not feature.get('geometry') or not feature.get('properties'):
            continue

        stats['total_locations'] += 1
        props = feature['properties']

        # Skip locations without proper site data (e.g., National Highways with empty fields)
        if props.get('defrasitename') == ' ' or props.get('status') != 'Current':
            continue

        exceeds_any = False
        exceeds_count = 0
        has_any_data = False

        # Process NO2
        no2_values = extract_pollutant_values(props, 'no2', years)
        if no2_values:
            has_any_data = True
            no2_avg = calculate_average(no2_values)
            feature['properties']['avg_no2_2020_2024'] = no2_avg
            feature['properties']['no2_years_with_data'] = len(no2_values)

            if no2_avg > WHO_GUIDELINES['NO2']:
                feature['properties']['exceeds_who_no2'] = True
                exceeds_any = True
                exceeds_count += 1
                stats['exceeding_no2'] += 1
            else:
                feature['properties']['exceeds_who_no2'] = False

        # Process PM2.5
        pm25_values = extract_pollutant_values(props, 'pm2_5', years)
        if pm25_values:
            has_any_data = True
            pm25_avg = calculate_average(pm25_values)
            feature['properties']['avg_pm2_5_2020_2024'] = pm25_avg
            feature['properties']['pm25_years_with_data'] = len(pm25_values)

            if pm25_avg > WHO_GUIDELINES['PM2.5']:
                feature['properties']['exceeds_who_pm2_5'] = True
                exceeds_any = True
                exceeds_count += 1
                stats['exceeding_pm25'] += 1
            else:
                feature['properties']['exceeds_who_pm2_5'] = False

        # Process PM10
        pm10_values = extract_pollutant_values(props, 'pm10', years)
        if pm10_values:
            has_any_data = True
            pm10_avg = calculate_average(pm10_values)
            feature['properties']['avg_pm10_2020_2024'] = pm10_avg
            feature['properties']['pm10_years_with_data'] = len(pm10_values)

            if pm10_avg > WHO_GUIDELINES['PM10']:
                feature['properties']['exceeds_who_pm10'] = True
                exceeds_any = True
                exceeds_count += 1
                stats['exceeding_pm10'] += 1
            else:
                feature['properties']['exceeds_who_pm10'] = False

        # Track statistics
        if has_any_data:
            stats['locations_with_data'] += 1

            # Add WHO guidelines to properties for reference
            feature['properties']['who_guideline_no2'] = WHO_GUIDELINES['NO2']
            feature['properties']['who_guideline_pm2_5'] = WHO_GUIDELINES['PM2.5']
            feature['properties']['who_guideline_pm10'] = WHO_GUIDELINES['PM10']

            if exceeds_any:
                filtered_features.append(feature)
                stats['locations_exceeding'] += 1

                if exceeds_count > 1:
                    stats['exceeding_multiple'] += 1

    # Prepare output data
    filtered_data = {
        'type': 'FeatureCollection',
        'name': 'RealTime_Monitors_WHO_Exceeding',
        'crs': data.get('crs'),
        'features': filtered_features,
        'metadata': {
            'source_file': os.path.basename(input_path),
            'years_analyzed': years,
            'who_guidelines': WHO_GUIDELINES,
            'statistics': stats
        }
    }

    # Save filtered data
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving file: {e}")
        return 0, len(data.get('features', [])), stats

    print(f"Real-Time Monitors: {stats['locations_exceeding']}/{stats['locations_with_data']} locations exceed at least one WHO guideline")
    print(f"  - Total locations: {stats['total_locations']}")
    print(f"  - Locations with valid data: {stats['locations_with_data']}")
    print(f"  - Exceeding NO2: {stats['exceeding_no2']}")
    print(f"  - Exceeding PM2.5: {stats['exceeding_pm25']}")
    print(f"  - Exceeding PM10: {stats['exceeding_pm10']}")
    print(f"  - Exceeding multiple pollutants: {stats['exceeding_multiple']}")

    return stats['locations_exceeding'], stats['total_locations'], stats

def main():
    """Main function with improved error handling and configuration."""
    # Configuration (could be moved to config file or command line args)
    base_path = "/Users/beomsu/Documents/ZeroDataChallenge"

    # File paths
    files = {
        'diffusion': {
            'input': os.path.join(base_path, "Air Quality Diffusion Tubes", "Air Quality Diffusion Tubes.geojson"),
            'output': os.path.join(base_path, "results", "Air_Quality_Diffusion_Tubes_WHO_Exceeding.geojson")
        },
        'monitors': {
            'input': os.path.join(base_path, "Air Quality Real-Time Monitors", "Air Quality Real-Time Monitors.geojson"),
            'output': os.path.join(base_path, "results", "Air_Quality_RealTime_Monitors_WHO_Exceeding.geojson")
        }
    }

    print("=" * 70)
    print("AIR QUALITY DATA PROCESSING - WHO GUIDELINE EXCEEDANCES (2020-2024)")
    print("=" * 70)
    print(f"WHO Guidelines:")
    print(f"  • NO2: {WHO_GUIDELINES['NO2']} µg/m³")
    print(f"  • PM2.5: {WHO_GUIDELINES['PM2.5']} µg/m³")
    print(f"  • PM10: {WHO_GUIDELINES['PM10']} µg/m³")
    print("-" * 70)

    # Process Diffusion Tubes
    if os.path.exists(files['diffusion']['input']):
        print("\n📍 Processing Diffusion Tubes (NO2 only)...")
        diff_filtered, diff_total, diff_stats = process_diffusion_tubes(
            files['diffusion']['input'],
            files['diffusion']['output']
        )
        if diff_filtered > 0 or diff_total > 0:
            print(f"✓ Saved to: {files['diffusion']['output']}")
    else:
        print(f"\n✗ Diffusion Tubes file not found: {files['diffusion']['input']}")

    # Process Real-Time Monitors
    if os.path.exists(files['monitors']['input']):
        print("\n📍 Processing Real-Time Monitors (NO2, PM2.5, PM10)...")
        mon_filtered, mon_total, mon_stats = process_realtime_monitors(
            files['monitors']['input'],
            files['monitors']['output']
        )
        if mon_filtered > 0 or mon_total > 0:
            print(f"✓ Saved to: {files['monitors']['output']}")
    else:
        print(f"\n✗ Real-Time Monitors file not found: {files['monitors']['input']}")

    print("-" * 70)
    print("Processing complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)