import base64
import json

import ee
import geemap.foliumap as geemap
import geopandas as gpd
import streamlit as st
from google.oauth2 import service_account

from utils.cache import load_cached_layer


def initialize_ee():
    """Initialize Earth Engine with service account credentials"""
    try:
        # Check if secrets are available
        if (
            "GOOGLE_SERVICE_ACCOUNT_KEY" in st.secrets
            and "GOOGLE_CLOUD_PROJECT_NAME" in st.secrets
        ):
            # Try to decode the key
            try:
                decoded = base64.b64decode(st.secrets["GOOGLE_SERVICE_ACCOUNT_KEY"])
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
            except Exception:
                # Fallback initialization
                ee.Initialize(project="ee-ciut")
        else:
            # Fallback initialization
            ee.Initialize(project="ee-ciut")

    except Exception as e:
        st.error(f"Earth Engine initialization failed: {str(e)}")
        raise


def load_basins():
    """Load the basins data"""
    basins_gdf = gpd.read_parquet("data/dipsoh_cuencas.parquet")
    # Convert to Earth Engine FeatureCollection
    basins_ee = geemap.geopandas_to_ee(basins_gdf)

    # Add area property to each feature
    basins_with_area = basins_ee.map(
        lambda feature: feature.set(
            {
                "area_m2": feature.geometry().area(),
                "area_km2": feature.geometry().area().divide(1e6),
                "basin_id": feature.get("cod_map"),
                "basin_name": feature.get("nombre"),
                "subregion": feature.get("subregion"),
                "subbasin": feature.get("sucuenca_n"),
            }
        )
    )

    return basins_with_area


def load_municipalities():
    """Load the municipalities data"""
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
    """Get area of interest (intersection of municipalities and basins)"""
    municipalities = load_municipalities()
    basins = load_basins()
    intersecting_basins = basins.filterBounds(municipalities.geometry())
    aoi = intersecting_basins.geometry()
    return aoi


def load_river_network(aoi):
    """Load river network within AOI"""
    river_atlas = ee.FeatureCollection(
        "projects/sat-io/open-datasets/HydroAtlas/RiverAtlas_v10"
    ).filterBounds(aoi)
    return river_atlas


def _load_population_from_gee(aoi):
    """Load population data for the AOI from GEE"""
    return (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filter(ee.Filter.eq("system:index", "ARG_2020"))
        .first()
        .clip(aoi)
    )


def load_population(aoi):
    """Load population data for the AOI with caching"""
    return load_cached_layer("population.tif", aoi, _load_population_from_gee)


def _load_impervious_surface_from_gee(aoi):
    """Load impervious surface data for the AOI from GEE"""
    return (
        ee.ImageCollection("projects/sat-io/open-datasets/GISA_1972_2019")
        .mosaic()
        .clip(aoi)
    )


def load_impervious_surface(aoi):
    """Load impervious surface data for the AOI with caching"""
    return load_cached_layer("impervious.tif", aoi, _load_impervious_surface_from_gee)


def _load_wetlands_from_gee(aoi):
    """Load wetlands data for the AOI from GEE"""
    return (
        ee.ImageCollection("projects/sat-io/open-datasets/GWL_FCS30")
        .sort("system:time_start", False)
        .first()
        .clip(aoi)
    )


def load_wetlands(aoi):
    """Load wetlands data for the AOI with caching"""
    return load_cached_layer("wetlands.tif", aoi, _load_wetlands_from_gee)


def _load_nighttime_lights_from_gee(aoi):
    """Load nighttime lights data for the AOI from GEE"""
    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
    start_date = "2020-01-01"
    end_date = "2021-01-01"
    filtered_viirs = viirs.filterDate(start_date, end_date)
    return filtered_viirs.qualityMosaic("avg_rad").clip(aoi)


def load_nighttime_lights(aoi):
    """Load nighttime lights data for the AOI with caching"""
    return load_cached_layer("nightlights.tif", aoi, _load_nighttime_lights_from_gee)


def calculate_vulnerability(population, ntl_composite, aoi):
    """Calculate vulnerability score based on population and nighttime lights"""
    # Add small constant to population to avoid division by zero
    # Only keep pixels with population > 0
    pop_mask = population.gt(0)
    population_masked = population.updateMask(pop_mask)
    safe_population = population_masked.add(0.1)

    # Calculate light per person ratio
    light_per_person = ntl_composite.divide(safe_population)

    # Apply log transformation to handle the skewed distribution
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


def vulnerability_analysis():
    """Main function for vulnerability analysis"""
    # Initialize Earth Engine
    initialize_ee()

    # Get area of interest and base data
    aoi = get_aoi()
    municipalities = load_municipalities()
    basins = load_basins().filterBounds(municipalities.geometry())

    # Create map with Carto Positron basemap
    m = geemap.Map(basemap="CartoDB.Positron")
    m.centerObject(basins, 9)

    # Set up visualization parameters
    population_vis = {
        "min": 0.0,
        "max": 50.0,
        "palette": ["24126c", "1fff4f", "d4ff50"],
    }

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

    vulnerability_vis = {
        "min": 0,
        "max": 1,
        "palette": ["green", "yellow", "orange", "red"],
    }

    # Style for geographic features
    municipalities_style = {"color": "#4C4E52", "fillColor": "00000000", "width": 1.5}
    basins_style = {"color": "black", "fillColor": "00000000", "width": 1}
    rivers_style = {"color": "blue", "width": 2}

    # Load all data
    styled_municipalities = municipalities.style(**municipalities_style)
    styled_basins = basins.style(**basins_style)
    river_network = load_river_network(aoi)
    styled_rivers = river_network.style(**rivers_style)

    population = load_population(aoi)
    impervious = load_impervious_surface(aoi)
    wetlands = load_wetlands(aoi)
    nightlights = load_nighttime_lights(aoi)
    vulnerability = calculate_vulnerability(population, nightlights, aoi)

    # Add all layers to the map
    # Analysis layers (bottom)
    m.addLayer(population, population_vis, "Población", False)
    m.addLayer(impervious, {}, "Superficies Impermeables", False)
    m.addLayer(wetlands, wetlands_vis, "Humedales", False)
    m.addLayer(vulnerability, vulnerability_vis, "Vulnerabilidad", True)

    # Reference layers (top)
    m.addLayer(styled_municipalities, {}, "Límites Municipales", False)
    m.addLayer(styled_basins, {}, "Cuencas", True)
    m.addLayer(styled_rivers, {}, "Red Fluvial", True)

    # Add legend for vulnerability
    legend_colors = ["#00ff00", "#ffff00", "#ffa500", "#ff0000"]
    legend_labels = ["Baja", "Media", "Alta", "Muy Alta"]
    m.add_legend(title="Vulnerabilidad", colors=legend_colors, labels=legend_labels)

    # Add a basemap selector
    m.add_basemap_selector()  # Simple version that includes all available basemaps

    # Add a layer manager - you've already included this
    m.add_layer_control()

    # Add an inspector tool
    m.add_inspector()  # This adds a tool to inspect pixel values when clicking

    # Display the map
    m.to_streamlit(height=600)
