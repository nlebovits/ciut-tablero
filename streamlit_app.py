import base64
import json
import os

import ee
import geemap.foliumap as geemap
import geopandas as gpd
import streamlit as st
from google.oauth2 import service_account
from PIL import Image

# Set page configuration
st.set_page_config(
    layout="wide", page_title="Riesgo de inundación en La Plata, Berisso y Ensenada"
)


# Load logo
logo_path = os.path.join("assets", "logo.png")
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.sidebar.image(logo, width=150)

# Sidebar information
st.sidebar.title("¿Qué?")
st.sidebar.info(
    """
    Esta aplicación analiza el riesgo de inundación en los municipios de La Plata, Berisso y Ensenada
    basándose en población, exposición, vulnerabilidad, superficies impermeables y humedales.
    """
)

st.sidebar.title("¿Quién?")
st.sidebar.info(
    """
    Hecho por [Nissim Lebovits](https://nlebovits.github.io/)
    """
)


@st.cache_resource
def initialize_ee():
    try:
        # Check if secrets are available
        if "GOOGLE_SERVICE_ACCOUNT_KEY" not in st.secrets:
            st.error("GOOGLE_SERVICE_ACCOUNT_KEY not found in secrets")
            raise KeyError("Missing service account key in secrets")

        if "GOOGLE_CLOUD_PROJECT_NAME" not in st.secrets:
            st.error("GOOGLE_CLOUD_PROJECT_NAME not found in secrets")
            raise KeyError("Missing cloud project name in secrets")

        # Try to decode the key
        try:
            decoded = base64.b64decode(st.secrets["GOOGLE_SERVICE_ACCOUNT_KEY"])

            # Parse JSON
            try:
                service_account_info = json.loads(decoded)

                # Create credentials
                credentials = service_account.Credentials.from_service_account_info(
                    service_account_info,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )

                # Initialize Earth Engine
                ee.Initialize(
                    credentials=credentials,
                    project=st.secrets["GOOGLE_CLOUD_PROJECT_NAME"],
                )

            except json.JSONDecodeError as e:
                st.error(f"Failed to parse JSON: {str(e)}")
                raise

        except Exception:
            # Try fallback method silently
            ee.Initialize(project="ee-ciut")

    except Exception as e:
        st.error(f"Earth Engine initialization failed: {str(e)}")
        raise


# Main function for vulnerability analysis
def vulnerability_analysis():
    row1_col1, row1_col2 = st.columns([3, 1])
    width = 950
    height = 600

    # Initialize map with Carto Positron basemap
    Map = geemap.Map(center=[-34.9, -57.95], zoom=10, basemap="CartoDB.Positron")

    # Define computation functions
    def get_municipalities():
        # Filter for target partidos in La Plata region
        target_partidos = ["La Plata", "Berisso", "Ensenada"]
        gaul_l2 = ee.FeatureCollection(
            "projects/sat-io/open-datasets/FAO/GAUL/GAUL_2024_L2"
        )
        municipalities = (
            gaul_l2.filter(ee.Filter.eq("gaul0_name", "Argentina"))
            .filter(ee.Filter.eq("gaul1_name", "Buenos Aires"))
            .filter(ee.Filter.inList("gaul2_name", target_partidos))
        )
        return municipalities

    def get_aoi():
        municipalities = get_municipalities()

        # Load basins from local parquet file
        basins_gdf = gpd.read_parquet("data/dipsoh_cuencas.parquet")

        # Convert GeoDataFrame to Earth Engine FeatureCollection
        basins_ee = geemap.geopandas_to_ee(basins_gdf)

        # Find intersecting basins
        intersecting_basins = basins_ee.filterBounds(municipalities.geometry())
        aoi = intersecting_basins.geometry()
        return aoi

    def get_population(aoi):
        # Filter just Argentina tiles (e.g., 2020 only)
        population = (
            ee.ImageCollection("WorldPop/GP/100m/pop")
            .filter(ee.Filter.eq("system:index", "ARG_2020"))
            .first()
            .clip(aoi)
        )
        return population

    def get_impervious_surface(aoi):
        # GISA Impervious Surface
        gisa = (
            ee.ImageCollection("projects/sat-io/open-datasets/GISA_1972_2019")
            .mosaic()
            .clip(aoi)
        )
        return gisa

    def get_wetlands(aoi):
        # Wetlands
        wetlands = (
            ee.ImageCollection("projects/sat-io/open-datasets/GWL_FCS30")
            .sort("system:time_start", False)
            .first()
            .clip(aoi)
        )
        return wetlands

    def get_river_network(aoi):
        river_atlas = ee.FeatureCollection(
            "projects/sat-io/open-datasets/HydroAtlas/RiverAtlas_v10"
        ).filterBounds(aoi)
        return river_atlas

    def get_nighttime_lights(aoi):
        # Nighttime lights with fixed year (2020)
        viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")

        # Fixed date range for 2020
        start_date = "2020-01-01"
        end_date = "2021-01-01"
        filtered_viirs = viirs.filterDate(start_date, end_date)
        ntl_composite = filtered_viirs.qualityMosaic("avg_rad").clip(aoi)

        return ntl_composite

    def calculate_pixel_vulnerability(population_img, nightlights_img):
        """
        Calculate vulnerability score at pixel level with log scaling
        Higher values indicate higher vulnerability
        Returns a single-band image for visualization with population mask
        """
        # Add small constant to population to avoid division by zero
        # Only keep pixels with population > 0
        pop_mask = population_img.gt(0)
        population_masked = population_img.updateMask(pop_mask)
        safe_population = population_masked.add(0.1)

        # Calculate light per person ratio
        light_per_person = nightlights_img.divide(safe_population)

        # Apply log transformation to handle the skewed distribution
        # Add small constant to avoid log(0)
        log_light_per_person = light_per_person.add(0.001).log()

        # Get percentiles for normalization
        percentiles = log_light_per_person.reduceRegion(
            reducer=ee.Reducer.percentile([2, 98]),
            geometry=aoi,
            scale=500,
            maxPixels=1e9,
        )

        # Extract min and max values (using 2nd and 98th percentiles for robustness)
        min_val = ee.Number(percentiles.get("avg_rad_p2"))
        max_val = ee.Number(percentiles.get("avg_rad_p98"))

        # Normalize to 0-1 scale
        normalized_lpp = log_light_per_person.subtract(min_val).divide(
            max_val.subtract(min_val)
        )

        # Clamp values to 0-1 range
        normalized_lpp = normalized_lpp.clamp(0, 1)

        # Invert scale so higher values = higher vulnerability
        vulnerability = ee.Image(1).subtract(normalized_lpp)

        # Make sure we have a single band named 'vulnerability'
        vulnerability = vulnerability.select([0], ["vulnerability"])

        # Fixed smoothing radius (500 meters)
        vulnerability_smoothed = vulnerability.focal_mean(radius=500, units="meters")

        return vulnerability_smoothed

    # Set up UI controls
    with row1_col2:
        # Group 2: Analysis layers (displayed first)
        analysis_layers = st.multiselect(
            "Capas de Análisis",
            [
                "Población",
                "Superficies Impermeables",
                "Humedales",
                "Vulnerabilidad",
            ],
            ["Vulnerabilidad"],
        )

        # Group 1: Geographic reference layers
        geo_layers = st.multiselect(
            "Capas de Referencia Geográfica",
            ["Límites Municipales", "Cuencas", "Red Fluvial"],
            ["Cuencas", "Red Fluvial"],
        )

        add_legend = st.checkbox("Mostrar leyenda de vulnerabilidad", True)

    # Compute data layers
    aoi = get_aoi()
    municipalities = get_municipalities()

    # Load basins from local parquet file for visualization
    basins_gdf = gpd.read_parquet("data/dipsoh_cuencas.parquet")
    basins = geemap.geopandas_to_ee(basins_gdf).filterBounds(municipalities.geometry())

    # First prepare and add all analysis layers (bottom layers)
    population = None
    if "Población" in analysis_layers or "Vulnerabilidad" in analysis_layers:
        population = get_population(aoi)
        if "Población" in analysis_layers:
            pop_vis = {
                "bands": ["population"],
                "min": 0.0,
                "max": 50.0,
                "palette": ["24126c", "1fff4f", "d4ff50"],
            }
            Map.addLayer(population, pop_vis, "Población")

    if "Superficies Impermeables" in analysis_layers:
        gisa = get_impervious_surface(aoi)
        # Using default visualization
        Map.addLayer(gisa, {}, "Superficies Impermeables")

    if "Humedales" in analysis_layers:
        wetlands = get_wetlands(aoi)
        wetlands_vis = {
            "min": 180,
            "max": 187,
            "palette": [
                "#CCCCCC",
                "#0000FF",
                "#006400",
                "#00FF00",
                "#00FFFF",
                "#CC99FF",
                "#556B2F",
                "#FFFF99",
                "#D2B48C",
            ],
        }
        Map.addLayer(wetlands.mask(wetlands.neq(0)), wetlands_vis, "Humedales")

    if "Vulnerabilidad" in analysis_layers:
        # Get nighttime lights with fixed parameters
        ntl_composite = get_nighttime_lights(aoi)

        # Calculate vulnerability
        vulnerability_smoothed = calculate_pixel_vulnerability(
            population, ntl_composite
        )

        vuln_vis = {"min": 0, "max": 1, "palette": ["green", "yellow", "orange", "red"]}
        Map.addLayer(vulnerability_smoothed, vuln_vis, "Vulnerabilidad")

        if add_legend:
            legend_colors = ["#00ff00", "#ffff00", "#ffa500", "#ff0000"]
            legend_labels = ["Baja", "Media", "Alta", "Muy Alta"]
            Map.add_legend(
                title="Vulnerabilidad", colors=legend_colors, labels=legend_labels
            )

    # Then prepare and add all geographic reference layers (top layers)
    if "Límites Municipales" in geo_layers:
        styled_municipalities = municipalities.style(
            **{"color": "#4C4E52", "fillColor": "00000000", "width": 1.5}
        )
        Map.addLayer(styled_municipalities, {}, "Límites Municipales")

    if "Cuencas" in geo_layers:
        styled_basins = basins.style(
            **{"color": "black", "fillColor": "00000000", "width": 1}
        )
        Map.addLayer(styled_basins, {}, "Cuencas")

    if "Red Fluvial" in geo_layers:
        river_atlas = get_river_network(aoi)
        styled_rivers = river_atlas.style(**{"color": "blue", "width": 2})
        Map.addLayer(styled_rivers, {}, "Red Fluvial")

    # Render map
    with row1_col1:
        Map.to_streamlit(width=width, height=height)


def app():
    # Initialize Earth Engine
    initialize_ee()

    st.title("Riesgo de inundación en La Plata, Berisso y Ensenada")

    # Run the main analysis directly without selection
    vulnerability_analysis()


if __name__ == "__main__":
    app()
