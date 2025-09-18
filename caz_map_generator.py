#!/usr/bin/env python3
"""
Clean Air Zone (CAZ) Visualization Script for Sheffield
This script analyzes air quality and traffic data to identify potential CAZ locations
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster, HeatMap
from shapely.geometry import Point
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import os

# Define paths
BASE_DIR = "/Users/beomsu/Documents/ZeroDataChallenge/results"
TRAFFIC_FILE = os.path.join(BASE_DIR, "Sheffield_Traffic_Top30_2020_2024.geojson")
POLLUTION_FILE = os.path.join(BASE_DIR, "Air_Quality_Diffusion_Tubes_Filtered_2020_2024.geojson")
OUTPUT_DIR = os.path.join(BASE_DIR, "Solution1")

def load_and_prepare_data():
    """Load and prepare traffic and pollution data"""
    print("Loading data...")
    traffic = gpd.read_file(TRAFFIC_FILE)
    pollution = gpd.read_file(POLLUTION_FILE)

    # Ensure WGS84 projection
    traffic = traffic.to_crs(epsg=4326)
    pollution = pollution.to_crs(epsg=4326)

    print(f"Loaded {len(traffic)} traffic points and {len(pollution)} pollution monitoring points")
    return traffic, pollution

def perform_clustering(traffic, pollution, n_clusters=3):
    """Perform K-means clustering to identify CAZ candidates"""
    print("Performing clustering analysis...")

    # Extract coordinates with weights
    # Higher weight for pollution data (0.6) vs traffic (0.4)
    traffic_points = np.array([[geom.x, geom.y] for geom in traffic.geometry])
    pollution_points = np.array([[geom.x, geom.y] for geom in pollution.geometry])

    # Combine all points for clustering
    all_points = np.vstack((traffic_points, pollution_points))

    # Weight array for visualization (not used in clustering directly)
    weights = np.concatenate([
        np.ones(len(traffic_points)) * 0.4,
        np.ones(len(pollution_points)) * 0.6
    ])

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(all_points)

    # Create cluster centers GeoDataFrame
    centers = [Point(xy) for xy in kmeans.cluster_centers_]
    clusters_gdf = gpd.GeoDataFrame(geometry=centers, crs="EPSG:4326")
    clusters_gdf["Cluster"] = [f"CAZ Candidate {i+1}" for i in range(n_clusters)]

    # Calculate cluster statistics and find nearby locations
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_points = all_points[cluster_mask]
        clusters_gdf.loc[i, "n_points"] = len(cluster_points)
        clusters_gdf.loc[i, "pollution_points"] = np.sum(cluster_mask[len(traffic_points):])
        clusters_gdf.loc[i, "traffic_points"] = np.sum(cluster_mask[:len(traffic_points)])

        # Find pollution sites in this cluster to identify location names
        pollution_in_cluster = pollution.iloc[cluster_mask[len(traffic_points):]]
        if len(pollution_in_cluster) > 0:
            # Get unique site names (remove duplicates and clean)
            site_names = pollution_in_cluster['defrasitename'].dropna().unique()
            # Take first 3 most common locations
            clusters_gdf.loc[i, "key_locations"] = ", ".join(site_names[:3]) if len(site_names) > 0 else "Unknown"

        # Find traffic locations in this cluster
        traffic_in_cluster = traffic.iloc[cluster_mask[:len(traffic_points)]]
        if len(traffic_in_cluster) > 0:
            road_names = traffic_in_cluster['road_name'].dropna().unique()
            clusters_gdf.loc[i, "major_roads"] = ", ".join(road_names[:3]) if len(road_names) > 0 else "N/A"

    return clusters_gdf, labels, all_points

def create_static_plot(traffic, pollution, clusters_gdf):
    """Create a static matplotlib plot with professional color scheme"""
    print("Creating static plot...")

    # Professional color palette with high contrast and accessibility
    # Based on best practices for data visualization
    COLOR_TRAFFIC = '#FF6B35'  # Vibrant coral-orange for traffic
    COLOR_POLLUTION_HIGH = '#8B0000'  # Dark red for high pollution
    COLOR_POLLUTION_LOW = '#2E7D32'  # Forest green for lower pollution
    COLOR_CAZ = '#1565C0'  # Strong blue for CAZ zones
    COLOR_CAZ_CIRCLE = '#4FC3F7'  # Light blue for zone boundaries

    fig, ax = plt.subplots(figsize=(14, 10), facecolor='#F5F5F5')
    ax.set_facecolor('white')

    # Plot traffic points with better visibility
    traffic.plot(ax=ax, color=COLOR_TRAFFIC, markersize=60, alpha=0.75,
                 label=f"Traffic Monitoring Points (n={len(traffic)})",
                 edgecolor='#C73E1D', linewidth=1.5, marker='s')  # Square markers

    # Plot pollution points with color coding based on NO2 levels
    high_pollution = pollution[pollution['avg_no2_2020_2024'] > 30] if 'avg_no2_2020_2024' in pollution.columns else pollution
    low_pollution = pollution[pollution['avg_no2_2020_2024'] <= 30] if 'avg_no2_2020_2024' in pollution.columns else gpd.GeoDataFrame()

    if len(high_pollution) > 0:
        high_pollution.plot(ax=ax, color=COLOR_POLLUTION_HIGH, markersize=45, alpha=0.8,
                           label=f"High NO2 (>30 µg/m³, n={len(high_pollution)})",
                           edgecolor='#4B0000', linewidth=1, marker='o')

    if len(low_pollution) > 0:
        low_pollution.plot(ax=ax, color=COLOR_POLLUTION_LOW, markersize=45, alpha=0.8,
                          label=f"Moderate NO2 (≤30 µg/m³, n={len(low_pollution)})",
                          edgecolor='#1B5E20', linewidth=1, marker='o')

    # Plot cluster centers with better visibility
    clusters_gdf.plot(ax=ax, color=COLOR_CAZ, markersize=250, marker="*",
                       label="CAZ Center Candidates", edgecolor='#0D47A1', linewidth=2)

    # Add circles around cluster centers (1.5 km radius approximation)
    for idx, row in clusters_gdf.iterrows():
        circle = plt.Circle((row.geometry.x, row.geometry.y), 0.015,
                           color=COLOR_CAZ_CIRCLE, fill=False, linestyle='--', linewidth=2.5, alpha=0.6)
        ax.add_patch(circle)

        # Add labels with location info
        label_text = f"{row['Cluster']}"
        if 'key_locations' in row and pd.notna(row['key_locations']):
            label_text += f"\n{row['key_locations'][:30]}..."  # Truncate if too long

        ax.annotate(label_text,
                   xy=(row.geometry.x, row.geometry.y),
                   xytext=(8, 8),
                   textcoords="offset points",
                   fontsize=9,
                   fontweight='bold',
                   color=COLOR_CAZ,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor=COLOR_CAZ))

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title("Sheffield Clean Air Zone (CAZ) Candidate Analysis\nBased on Traffic Volume and Air Quality Data",
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "caz_static_map.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Static plot saved to {output_path}")
    plt.close()

def create_interactive_map(traffic, pollution, clusters_gdf):
    """Create an interactive Folium map"""
    print("Creating interactive map...")

    # Calculate map center
    all_lats = list(traffic.geometry.y) + list(pollution.geometry.y)
    all_lons = list(traffic.geometry.x) + list(pollution.geometry.x)
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)

    # Create base map
    m = folium.Map(location=[center_lat, center_lon],
                   zoom_start=11,
                   tiles="OpenStreetMap",
                   prefer_canvas=True)

    # Add tile layers
    folium.TileLayer('cartodbpositron', name='Light Map').add_to(m)
    folium.TileLayer('cartodbdark_matter', name='Dark Map').add_to(m)

    # Create feature groups for different layers
    traffic_group = folium.FeatureGroup(name="Traffic Monitoring Points")
    pollution_group = folium.FeatureGroup(name="Air Quality Monitoring Points")
    caz_group = folium.FeatureGroup(name="CAZ Candidates")

    # Professional colors for interactive map
    COLOR_TRAFFIC_WEB = '#FF6B35'
    COLOR_POLLUTION_HIGH_WEB = '#DC143C'  # Crimson for high pollution
    COLOR_POLLUTION_MED_WEB = '#FF8C00'  # Dark orange for medium
    COLOR_POLLUTION_LOW_WEB = '#32CD32'  # Lime green for low
    COLOR_CAZ_WEB = '#1E90FF'  # Dodger blue

    # Add traffic points
    for idx, row in traffic.iterrows():
        year = row.get('year', 'N/A')
        count = row.get('all_motor_vehicles', 'N/A')
        road = row.get('road_name', 'Unknown')

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=7,
            popup=folium.Popup(
                f"<b>🚗 Traffic Monitoring Point</b><br>"
                f"<b>Road:</b> {road}<br>"
                f"<b>Year:</b> {year}<br>"
                f"<b>Vehicle Count:</b> {count:,}" if count != 'N/A' else f"<b>Vehicle Count:</b> {count}",
                max_width=250
            ),
            color=COLOR_TRAFFIC_WEB,
            fill=True,
            fillColor=COLOR_TRAFFIC_WEB,
            fillOpacity=0.75,
            weight=2
        ).add_to(traffic_group)

    # Add pollution points with color gradient based on NO2 levels
    for idx, row in pollution.iterrows():
        site_name = row.get('defrasitename', 'Unknown')
        no2_2024 = row.get('no2_2024', 'N/A')
        avg_no2 = row.get('avg_no2_2020_2024', 'N/A')
        exceeds = row.get('exceeds_who_no2', False)

        # Color based on pollution level
        if avg_no2 != 'N/A' and avg_no2 > 40:
            color = COLOR_POLLUTION_HIGH_WEB
            level = "High"
        elif avg_no2 != 'N/A' and avg_no2 > 25:
            color = COLOR_POLLUTION_MED_WEB
            level = "Medium"
        else:
            color = COLOR_POLLUTION_LOW_WEB
            level = "Low"

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=6,
            popup=folium.Popup(
                f"<b>🌬️ Air Quality Monitoring Site</b><br>"
                f"<b>Location:</b> {site_name}<br>"
                f"<b>2024 NO2:</b> {no2_2024:.1f} µg/m³<br>" if no2_2024 != 'N/A' else f"<b>2024 NO2:</b> {no2_2024}<br>"
                f"<b>5-Year Avg:</b> {avg_no2:.1f} µg/m³<br>" if avg_no2 != 'N/A' else f"<b>5-Year Avg:</b> {avg_no2}<br>"
                f"<b>Pollution Level:</b> {level}<br>"
                f"<b>Exceeds WHO:</b> {'⚠️ Yes' if exceeds else '✅ No'}",
                max_width=300
            ),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.8,
            weight=2
        ).add_to(pollution_group)

    # Add CAZ candidate zones
    for idx, row in clusters_gdf.iterrows():
        # Build popup content with location info
        popup_content = f"<b>⭐ {row['Cluster']}</b><br><br>"
        if 'key_locations' in row and pd.notna(row['key_locations']):
            popup_content += f"<b>Key Areas:</b> {row['key_locations']}<br>"
        if 'major_roads' in row and pd.notna(row['major_roads']):
            popup_content += f"<b>Major Roads:</b> {row['major_roads']}<br>"
        popup_content += f"<br><b>Monitoring Points:</b><br>"
        popup_content += f"• Traffic: {int(row['traffic_points'])}<br>"
        popup_content += f"• Air Quality: {int(row['pollution_points'])}<br>"
        popup_content += f"• Total: {int(row['n_points'])}"

        # Add center marker
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=folium.Popup(popup_content, max_width=350),
            icon=folium.Icon(color="blue", icon="star", prefix='fa'),
            tooltip=f"{row['Cluster']}"
        ).add_to(caz_group)

        # Add zone circle (approximately 1.5 km radius)
        folium.Circle(
            location=[row.geometry.y, row.geometry.x],
            radius=1500,  # 1.5 km radius
            popup=f"{row['Cluster']} Zone (1.5km radius)",
            color=COLOR_CAZ_WEB,
            fill=True,
            fillColor=COLOR_CAZ_WEB,
            fillOpacity=0.15,
            weight=3,
            dashArray='10, 5'
        ).add_to(caz_group)

    # Add groups to map
    traffic_group.add_to(m)
    pollution_group.add_to(m)
    caz_group.add_to(m)

    # Add heatmap for pollution intensity
    heat_data = [[row.geometry.y, row.geometry.x, row.get('avg_no2_2020_2024', 20)]
                 for idx, row in pollution.iterrows() if row.get('avg_no2_2020_2024')]

    if heat_data:
        HeatMap(heat_data,
                name="NO2 Concentration Heatmap",
                min_opacity=0.3,
                max_zoom=18,
                radius=20,
                blur=15,
                gradient={0.0: 'green', 0.5: 'yellow', 1.0: 'red'}).add_to(m)

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Add title
    title_html = '''
                 <div style="position: fixed;
                            top: 10px; left: 50%; transform: translateX(-50%);
                            width: 600px; height: 60px;
                            background-color: white; border-radius: 10px;
                            border: 2px solid grey; z-index: 9999;
                            font-size: 16px; font-weight: bold; text-align: center;
                            padding-top: 20px; opacity: 0.9">
                 Sheffield Clean Air Zone (CAZ) Candidate Analysis
                 </div>
                 '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Save map
    output_path = os.path.join(OUTPUT_DIR, "caz_interactive_map.html")
    m.save(output_path)
    print(f"Interactive map saved to {output_path}")

    return m

def generate_summary_report(traffic, pollution, clusters_gdf):
    """Generate a summary report of the analysis"""
    print("Generating summary report...")

    report = []
    report.append("=" * 70)
    report.append("SHEFFIELD CLEAN AIR ZONE (CAZ) ANALYSIS REPORT")
    report.append("=" * 70)
    report.append("")
    report.append("DATA SUMMARY:")
    report.append(f"- Total Traffic Monitoring Points: {len(traffic)}")
    report.append(f"- Total Air Quality Monitoring Points: {len(pollution)}")
    report.append("")

    # Pollution statistics
    exceeds_count = pollution['exceeds_who_no2'].sum() if 'exceeds_who_no2' in pollution.columns else 0
    avg_no2 = pollution['avg_no2_2020_2024'].mean() if 'avg_no2_2020_2024' in pollution.columns else 0

    report.append("AIR QUALITY STATISTICS:")
    report.append(f"- Sites exceeding WHO NO2 guideline: {exceeds_count}/{len(pollution)}")
    report.append(f"- Average NO2 concentration (2020-2024): {avg_no2:.2f} µg/m³")
    report.append(f"- WHO guideline: 10 µg/m³")
    report.append("")

    report.append("CAZ CANDIDATE ZONES:")
    report.append("-" * 50)
    for idx, row in clusters_gdf.iterrows():
        report.append(f"\n{row['Cluster']}:")
        report.append(f"  Coordinates: ({row.geometry.y:.6f}, {row.geometry.x:.6f})")

        # Add location information
        if 'key_locations' in row and pd.notna(row['key_locations']):
            report.append(f"  Key Areas: {row['key_locations']}")
        if 'major_roads' in row and pd.notna(row['major_roads']):
            report.append(f"  Major Roads: {row['major_roads']}")

        report.append(f"  Monitoring Points Statistics:")
        report.append(f"    - Traffic monitoring points: {int(row['traffic_points'])}")
        report.append(f"    - Air quality monitoring points: {int(row['pollution_points'])}")
        report.append(f"    - Total points in cluster: {int(row['n_points'])}")

    report.append("\n" + "=" * 70)
    report.append("Analysis complete. Results saved in Solution1 folder.")
    report.append("=" * 70)

    # Save report
    report_text = "\n".join(report)
    report_path = os.path.join(OUTPUT_DIR, "analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to {report_path}")

def main():
    """Main execution function"""
    print("Starting Clean Air Zone Analysis for Sheffield...")
    print("=" * 50)

    # Load data
    traffic, pollution = load_and_prepare_data()

    # Perform clustering
    clusters_gdf, labels, all_points = perform_clustering(traffic, pollution, n_clusters=3)

    # Create visualizations
    create_static_plot(traffic, pollution, clusters_gdf)
    create_interactive_map(traffic, pollution, clusters_gdf)

    # Generate report
    generate_summary_report(traffic, pollution, clusters_gdf)

    print("\n✅ Analysis complete! Check the Solution1 folder for results:")
    print(f"   - caz_interactive_map.html : Interactive web map")
    print(f"   - caz_static_map.png : Static visualization")
    print(f"   - analysis_report.txt : Summary report")

if __name__ == "__main__":
    main()