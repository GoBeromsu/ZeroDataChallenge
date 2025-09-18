#!/usr/bin/env python3
"""
Sheffield Clean Air Zone (CAZ) Dashboard
Interactive visualization for air quality and traffic analysis
"""

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from shapely.geometry import Point
import os

# Page config
st.set_page_config(
    page_title="Sheffield CAZ Analysis",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling and larger components
st.markdown(
    """
    <style>
    .main {
        padding: 0rem 0rem;
        max-width: 100%;
    }
    .stAlert {
        background-color: #f0f2f6;
        border: 2px solid #1E90FF;
        font-size: 1.1rem;
        padding: 1.5rem;
    }
    h1 {
        color: #1565C0;
        font-size: 3rem !important;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    h2 {
        font-size: 2.2rem !important;
        color: #2E7D32;
        margin-top: 2rem;
    }
    h3 {
        font-size: 1.8rem !important;
        color: #424242;
    }
    h4 {
        font-size: 1.5rem !important;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 30px;
        border-radius: 15px;
        border-left: 8px solid #1E90FF;
        margin-bottom: 20px;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .stMetric > div {
        font-size: 1.3rem !important;
    }
    .stMetric [data-testid="metric-container"] > div:nth-child(1) {
        font-size: 1rem !important;
        font-weight: 600;
        color: #666;
    }
    .stMetric [data-testid="metric-container"] > div:nth-child(2) {
        font-size: 2.5rem !important;
        font-weight: bold;
    }
    .stMetric [data-testid="metric-container"] > div:nth-child(3) {
        font-size: 1.1rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.3rem !important;
        font-weight: 600;
        padding: 15px 30px;
    }
    .stExpander {
        font-size: 1.2rem !important;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .stButton > button {
        font-size: 1.2rem !important;
        padding: 12px 30px;
        font-weight: 600;
    }
    .stSelectbox > label, .stMultiSelect > label, .stSlider > label {
        font-size: 1.2rem !important;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .stCheckbox > label {
        font-size: 1.15rem !important;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        padding: 2rem 1rem;
    }
    div[data-testid="stSidebar"] h2 {
        color: #1565C0;
        border-bottom: 3px solid #1E90FF;
        padding-bottom: 10px;
    }
    .plot-container {
        padding: 20px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    p {
        font-size: 1.1rem;
        line-height: 1.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.title("🌍 Sheffield Clean Air Zone (CAZ) Analysis Dashboard")
st.markdown(
    """
    **Interactive visualization of potential Clean Air Zones based on traffic volume and air quality monitoring data**
"""
)


# Load data
@st.cache_data
def load_data():
    """Load and cache the geospatial data"""
    # Use relative path that works both locally and on Streamlit Cloud
    current_dir = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.join(current_dir, "results")

    # Define file paths
    traffic_file = os.path.join(BASE_DIR, "Sheffield_Traffic_Top30_2020_2024.geojson")
    pollution_file = os.path.join(BASE_DIR, "Air_Quality_Diffusion_Tubes_Filtered_2020_2024.geojson")

    # Check if files exist and provide helpful error messages
    if not os.path.exists(BASE_DIR):
        st.error(f"Results directory not found. Looking for: {BASE_DIR}")
        st.info("Please ensure the 'results' folder is in the same directory as streamlit_app.py")
        st.stop()

    if not os.path.exists(traffic_file):
        st.error(f"Traffic data file not found: Sheffield_Traffic_Top30_2020_2024.geojson")
        st.info(f"Looking in: {BASE_DIR}")
        available_files = os.listdir(BASE_DIR) if os.path.exists(BASE_DIR) else []
        if available_files:
            st.info(f"Available files in results folder: {', '.join(available_files)}")
        st.stop()

    if not os.path.exists(pollution_file):
        st.error(f"Pollution data file not found: Air_Quality_Diffusion_Tubes_Filtered_2020_2024.geojson")
        st.info(f"Looking in: {BASE_DIR}")
        st.stop()

    # Load the data
    try:
        traffic = gpd.read_file(traffic_file)
        pollution = gpd.read_file(pollution_file)
    except Exception as e:
        st.error(f"Error loading data files: {str(e)}")
        st.info("Please check that the GeoJSON files are valid and not corrupted.")
        st.stop()

    # Ensure WGS84 projection
    traffic = traffic.to_crs(epsg=4326)
    pollution = pollution.to_crs(epsg=4326)

    return traffic, pollution


def perform_clustering(traffic, pollution, n_clusters=3):
    """Perform K-means clustering to identify CAZ candidates"""

    # Extract coordinates
    traffic_points = np.array([[geom.x, geom.y] for geom in traffic.geometry])
    pollution_points = np.array([[geom.x, geom.y] for geom in pollution.geometry])

    # Combine all points
    all_points = np.vstack((traffic_points, pollution_points))

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(all_points)

    # Create cluster centers GeoDataFrame
    centers = [Point(xy) for xy in kmeans.cluster_centers_]
    clusters_gdf = gpd.GeoDataFrame(geometry=centers, crs="EPSG:4326")
    clusters_gdf["Cluster"] = [f"CAZ Zone {i+1}" for i in range(n_clusters)]

    # Calculate statistics for each cluster
    for i in range(n_clusters):
        cluster_mask = labels == i
        clusters_gdf.loc[i, "n_points"] = np.sum(cluster_mask)
        clusters_gdf.loc[i, "pollution_points"] = np.sum(
            cluster_mask[len(traffic_points) :]
        )
        clusters_gdf.loc[i, "traffic_points"] = np.sum(
            cluster_mask[: len(traffic_points)]
        )

        # Find key locations
        pollution_in_cluster = pollution.iloc[cluster_mask[len(traffic_points) :]]
        if len(pollution_in_cluster) > 0:
            site_names = pollution_in_cluster["defrasitename"].dropna().unique()
            clusters_gdf.loc[i, "key_locations"] = (
                ", ".join(site_names[:3]) if len(site_names) > 0 else "Unknown"
            )

        traffic_in_cluster = traffic.iloc[cluster_mask[: len(traffic_points)]]
        if len(traffic_in_cluster) > 0:
            road_names = traffic_in_cluster["road_name"].dropna().unique()
            clusters_gdf.loc[i, "major_roads"] = (
                ", ".join(road_names[:3]) if len(road_names) > 0 else "N/A"
            )

    return clusters_gdf, labels


# Load data
traffic, pollution = load_data()

# Sidebar
with st.sidebar:
    st.header("⚙️ Analysis Settings")

    # Number of CAZ zones
    n_zones = st.slider(
        "Number of CAZ Zones",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of potential Clean Air Zones to identify",
    )

    # Map style
    st.subheader("🗺️ Map Settings")
    map_style = st.selectbox(
        "Map Style",
        options=["OpenStreetMap", "CartoDB Positron", "CartoDB Dark Matter"],
        index=1,
    )

    # Data period info
    st.markdown("---")
    st.info(
        """
    **📅 Data Period**

    All data shown: **2020-2024**
    - Traffic monitoring data
    - Air quality measurements (NO2)
    - 5-year averages calculated
    """
    )

# Use all data without filtering
traffic_filtered = traffic
pollution_filtered = pollution

# Perform clustering
clusters_gdf, labels = perform_clustering(traffic_filtered, pollution_filtered, n_zones)

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(
    ["📍 Interactive Map", "📊 Statistics", "📈 Analysis", "📝 Report"]
)

with tab1:
    st.header("Interactive CAZ Map")

    # Create columns for controls
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        show_traffic = st.checkbox("🚗 Traffic", value=True)
    with col2:
        show_pollution = st.checkbox("🌬️ Air Quality", value=True)
    with col3:
        show_heatmap = st.checkbox("🔥 Heatmap", value=False)
    with col4:
        point_size = st.slider("Point Size", 3, 15, 7, help="Adjust marker size")
    with col5:
        zone_radius = st.slider(
            "Zone Radius (km)", 0.5, 3.0, 1.5, 0.1, help="CAZ zone radius"
        )

    # Create folium map
    center_lat = np.mean([geom.y for geom in pollution.geometry])
    center_lon = np.mean([geom.x for geom in pollution.geometry])

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles=(
            map_style.replace(" ", "").lower()
            if map_style != "OpenStreetMap"
            else "OpenStreetMap"
        ),
    )

    # Add traffic points with adjustable size
    if show_traffic:
        for idx, row in traffic_filtered.iterrows():
            vehicles = row.get("all_motor_vehicles", "N/A")
            road = row.get("road_name", "Unknown")
            year = row.get("year", "N/A")

            # Format vehicle count properly
            vehicles_text = f"{vehicles:,}" if isinstance(vehicles, (int, float)) and vehicles != 'N/A' else str(vehicles)

            popup_text = f"""
            <div style='font-size: 14px;'>
                <b>🚗 Traffic Monitoring Point</b><br>
                <b>Road:</b> {road}<br>
                <b>Year:</b> {year}<br>
                <b>Total Vehicles:</b> {vehicles_text}
            </div>
            """

            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=point_size,
                popup=folium.Popup(popup_text, max_width=300),
                color="#FF6B35",
                fill=True,
                fillOpacity=0.7,
                weight=2,
            ).add_to(m)

    # Add pollution points with adjustable size
    if show_pollution:
        for idx, row in pollution_filtered.iterrows():
            avg_no2 = row.get("avg_no2_2020_2024", 0)
            site_name = row.get("defrasitename", "Unknown")
            no2_2024 = row.get("no2_2024", "N/A")

            if avg_no2 > 40:
                color = "#DC143C"
                level = "High"
            elif avg_no2 > 25:
                color = "#FF8C00"
                level = "Medium"
            else:
                color = "#32CD32"
                level = "Low"

            # Format NO2 values properly
            no2_2024_text = f"{no2_2024:.1f} µg/m³" if isinstance(no2_2024, (int, float)) and no2_2024 != 'N/A' else str(no2_2024)

            popup_text = f"""
            <div style='font-size: 14px;'>
                <b>🌬️ Air Quality Site</b><br>
                <b>Location:</b> {site_name}<br>
                <b>2024 NO2:</b> {no2_2024_text}<br>
                <b>5-Year Avg:</b> {avg_no2:.1f} µg/m³<br>
                <b>Level:</b> <span style='color: {color}'>{level}</span>
            </div>
            """

            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=point_size - 1,  # Slightly smaller than traffic points
                popup=folium.Popup(popup_text, max_width=350),
                color=color,
                fill=True,
                fillOpacity=0.8,
                weight=2,
            ).add_to(m)

    # Add CAZ zones with adjustable radius
    for idx, row in clusters_gdf.iterrows():
        popup_text = f"""
        <div style='font-size: 14px;'>
            <b>⭐ {row['Cluster']}</b><br>
            <b>Key Areas:</b> {row.get('key_locations', 'Unknown')}<br>
            <b>Major Roads:</b> {row.get('major_roads', 'N/A')}<br>
            <b>Monitoring Points:</b> {int(row['n_points'])}
        </div>
        """

        # Add center marker
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=folium.Popup(popup_text, max_width=400),
            icon=folium.Icon(color="blue", icon="star", prefix="fa"),
            tooltip=f"{row['Cluster']}",
        ).add_to(m)

        # Add zone circle with adjustable radius
        folium.Circle(
            location=[row.geometry.y, row.geometry.x],
            radius=zone_radius * 1000,  # Convert km to meters
            color="#1E90FF",
            fill=True,
            fillOpacity=0.15,
            weight=3,
            dashArray="10, 5",
        ).add_to(m)

    # Add heatmap
    if show_heatmap and "avg_no2_2020_2024" in pollution.columns:
        heat_data = [
            [row.geometry.y, row.geometry.x, row["avg_no2_2020_2024"]]
            for idx, row in pollution_filtered.iterrows()
        ]
        HeatMap(
            heat_data,
            radius=20,
            blur=15,
            gradient={0.0: "green", 0.5: "yellow", 1.0: "red"},
        ).add_to(m)

    # Display map with larger size
    st_folium(m, width=None, height=800, returned_objects=[], use_container_width=True)

with tab2:
    st.header("📊 Key Statistics")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Traffic Points",
            len(traffic_filtered),
            (
                f"{len(traffic_filtered) - len(traffic)} filtered"
                if len(traffic_filtered) < len(traffic)
                else None
            ),
        )

    with col2:
        st.metric(
            "Air Quality Sites",
            len(pollution_filtered),
            (
                f"{len(pollution_filtered) - len(pollution)} filtered"
                if len(pollution_filtered) < len(pollution)
                else None
            ),
        )

    with col3:
        avg_no2 = (
            pollution["avg_no2_2020_2024"].mean()
            if "avg_no2_2020_2024" in pollution.columns
            else 0
        )
        st.metric(
            "Avg NO2 (µg/m³)",
            f"{avg_no2:.1f}",
            f"{avg_no2 - 10:.1f} above WHO" if avg_no2 > 10 else "Within WHO limits",
            delta_color="inverse",
        )

    with col4:
        exceeds = (
            pollution["exceeds_who_no2"].sum()
            if "exceeds_who_no2" in pollution.columns
            else 0
        )
        st.metric(
            "Sites > WHO Limit",
            exceeds,
            f"{(exceeds/len(pollution)*100):.0f}% of sites",
        )

    # CAZ Zone details
    st.subheader("🎯 CAZ Zone Details")

    for idx, row in clusters_gdf.iterrows():
        with st.expander(
            f"{row['Cluster']} - {row.get('key_locations', 'Unknown Area')[:50]}..."
        ):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"""
                **📍 Location Details:**
                - Coordinates: ({row.geometry.y:.4f}, {row.geometry.x:.4f})
                - Key Areas: {row.get('key_locations', 'N/A')}
                - Major Roads: {row.get('major_roads', 'N/A')}
                """
                )
            with col2:
                st.markdown(
                    f"""
                **📊 Monitoring Points:**
                - Traffic Points: {int(row['traffic_points'])}
                - Air Quality Points: {int(row['pollution_points'])}
                - Total Points: {int(row['n_points'])}
                """
                )

with tab3:
    st.header("📈 Data Analysis")

    # NO2 Distribution
    st.subheader("NO2 Concentration Distribution")

    if "avg_no2_2020_2024" in pollution.columns:
        fig = px.histogram(
            pollution,
            x="avg_no2_2020_2024",
            nbins=30,
            title="Distribution of NO2 Concentrations (2020-2024 Average)",
            labels={
                "avg_no2_2020_2024": "NO2 Concentration (µg/m³)",
                "count": "Number of Sites",
            },
            color_discrete_sequence=["#1E90FF"],
        )
        fig.add_vline(
            x=10, line_dash="dash", line_color="red", annotation_text="WHO Limit"
        )
        fig.add_vline(
            x=40, line_dash="dash", line_color="orange", annotation_text="UK Limit"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Traffic volume by road
    st.subheader("Traffic Volume by Road")

    if "road_name" in traffic.columns and "all_motor_vehicles" in traffic.columns:
        traffic_by_road = (
            traffic_filtered.groupby("road_name")["all_motor_vehicles"]
            .sum()
            .sort_values(ascending=True)
            .tail(10)
        )

        fig = px.bar(
            x=traffic_by_road.values,
            y=traffic_by_road.index,
            orientation="h",
            title="Top 10 Roads by Traffic Volume",
            labels={"x": "Total Vehicle Count", "y": "Road"},
            color_discrete_sequence=["#FF6B35"],
        )
        st.plotly_chart(fig, use_container_width=True)

    # Time series if available
    if "year" in traffic.columns:
        st.subheader("Traffic Trends Over Time")

        traffic_yearly = traffic.groupby("year")["all_motor_vehicles"].agg(
            ["sum", "mean"]
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=traffic_yearly.index,
                y=traffic_yearly["sum"],
                mode="lines+markers",
                name="Total Traffic",
                line=dict(color="#FF6B35", width=3),
                marker=dict(size=10),
            )
        )

        fig.update_layout(
            title="Traffic Volume Trend",
            xaxis_title="Year",
            yaxis_title="Total Vehicle Count",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("📝 Analysis Report")

    # Generate report summary
    st.markdown(
        """
    ### Executive Summary

    This analysis identifies optimal locations for Clean Air Zones (CAZ) in Sheffield based on:
    - **Traffic monitoring data** from major roads and junctions
    - **Air quality measurements** from NO2 diffusion tubes across the city
    - **K-means clustering** to identify zones with highest impact potential
    """
    )

    # Zone recommendations
    st.subheader("🎯 Recommended CAZ Zones")

    for idx, row in clusters_gdf.iterrows():
        st.markdown(
            f"""
        #### {row['Cluster']}
        - **Location**: {row.get('key_locations', 'Unknown')}
        - **Major Roads**: {row.get('major_roads', 'N/A')}
        - **Coverage**: {int(row['pollution_points'])} air quality sites, {int(row['traffic_points'])} traffic points
        - **Coordinates**: {row.geometry.y:.6f}, {row.geometry.x:.6f}
        """
        )

    # Key findings
    st.subheader("📊 Key Findings")

    if "avg_no2_2020_2024" in pollution.columns:
        avg_no2 = pollution["avg_no2_2020_2024"].mean()
        exceeds = (
            pollution["exceeds_who_no2"].sum()
            if "exceeds_who_no2" in pollution.columns
            else 0
        )

        st.info(
            f"""
        - **Average NO2 Concentration**: {avg_no2:.1f} µg/m³ (WHO limit: 10 µg/m³)
        - **Sites Exceeding WHO Guidelines**: {exceeds}/{len(pollution)} ({exceeds/len(pollution)*100:.0f}%)
        - **Identified CAZ Zones**: {n_zones}
        - **Total Monitoring Points**: {len(traffic) + len(pollution)}
        """
        )

    # Download button for report
    report_text = f"""
Sheffield Clean Air Zone Analysis Report
========================================

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

SUMMARY STATISTICS:
- Total Traffic Monitoring Points: {len(traffic)}
- Total Air Quality Monitoring Points: {len(pollution)}
- Average NO2 Concentration: {pollution['avg_no2_2020_2024'].mean():.1f} µg/m³
- Sites Exceeding WHO Guidelines: {pollution['exceeds_who_no2'].sum() if 'exceeds_who_no2' in pollution.columns else 0}

RECOMMENDED CAZ ZONES:
"""

    for idx, row in clusters_gdf.iterrows():
        report_text += f"""
{row['Cluster']}:
  Location: {row.get('key_locations', 'Unknown')}
  Major Roads: {row.get('major_roads', 'N/A')}
  Coordinates: ({row.geometry.y:.6f}, {row.geometry.x:.6f})
  Monitoring Points: {int(row['n_points'])} total
"""

    st.download_button(
        label="📥 Download Report",
        data=report_text,
        file_name=f"sheffield_caz_report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
    )

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center'>
    <p>🌍 Sheffield Clean Air Zone Analysis | Data: 2020-2024 |
    <a href='https://github.com/yourusername/sheffield-caz'>GitHub</a></p>
</div>
""",
    unsafe_allow_html=True,
)
