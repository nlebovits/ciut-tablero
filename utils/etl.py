"""ETL utilities for CIUT Tablero project.

This module provides functions for extracting, transforming, and loading
geospatial data from Earth Engine to Google Cloud Storage and back.
"""

import logging
import re
import time
from typing import Dict, List, Optional, Set

import ee
import geemap
import geopandas as gpd
from google.cloud import storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project="ee-ciut")

# Constants
BUCKET_NAME = "ciut-tablero"
GOOGLE_CLOUD_PROJECT = "ee-ciut"


def data_exists(bucket_name: str, prefix: str) -> bool:
    """Check if files exist in a GCS bucket.

    Args:
        bucket_name: Name of the GCS bucket
        prefix: Prefix to check for files

    Returns:
        bool: True if files exist, False otherwise
    """
    try:
        storage_client = storage.Client(project=GOOGLE_CLOUD_PROJECT)
        bucket = storage_client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))
        return len(blobs) > 0
    except Exception as e:
        logger.error(f"Error checking bucket {bucket_name}: {e}")
        return False


def list_and_check_gcs_files(bucket_name: str, prefix: str) -> List[str]:
    """Check if files exist in a GCS bucket folder and list them if they do.

    Args:
        bucket_name: Name of the GCS bucket
        prefix: Prefix to check for files

    Returns:
        List[str]: List of GCS file paths
    """
    try:
        client = storage.Client(project=GOOGLE_CLOUD_PROJECT)
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))

        if not blobs:
            logger.warning(
                f"No files found with prefix '{prefix}' in bucket '{bucket_name}'"
            )
            return []

        file_urls = [
            f"gs://{bucket_name}/{blob.name}"
            for blob in blobs
            if blob.name.endswith(".tif")
        ]
        return file_urls
    except Exception as e:
        logger.error(f"Error accessing bucket {bucket_name}: {e}")
        return []


def extract_date_from_filename(filename: str) -> Optional[str]:
    """Extract date from filename in YYYY-MM-DD format.

    Args:
        filename: Filename to extract date from

    Returns:
        Optional[str]: Extracted date or None if not found
    """
    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
    return match.group(0) if match else None


def get_municipalities() -> ee.FeatureCollection:
    """Get municipalities in La Plata region.

    Returns:
        ee.FeatureCollection: Feature collection of municipalities
    """
    target_partidos = ["La Plata", "Berisso", "Ensenada"]
    gaul_l2 = ee.FeatureCollection(
        "projects/sat-io/open-datasets/FAO/GAUL/GAUL_2024_L2"
    )
    return (
        gaul_l2.filter(ee.Filter.eq("gaul0_name", "Argentina"))
        .filter(ee.Filter.eq("gaul1_name", "Buenos Aires"))
        .filter(ee.Filter.inList("gaul2_name", target_partidos))
    )


def get_population(aoi: ee.Geometry) -> ee.Image:
    """Get population data for Argentina.

    Args:
        aoi: Area of interest geometry

    Returns:
        ee.Image: Population image
    """
    return (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filter(ee.Filter.eq("system:index", "ARG_2020"))
        .first()
        .clip(aoi)
    )


def get_impervious_surface(aoi: ee.Geometry) -> ee.Image:
    """Get GISA impervious surface data.

    Args:
        aoi: Area of interest geometry

    Returns:
        ee.Image: Impervious surface image
    """
    return (
        ee.ImageCollection("projects/sat-io/open-datasets/GISA_1972_2019")
        .mosaic()
        .clip(aoi)
    )


def get_wetlands(aoi: ee.Geometry) -> ee.Image:
    """Get wetlands data.

    Args:
        aoi: Area of interest geometry

    Returns:
        ee.Image: Wetlands image
    """
    return (
        ee.ImageCollection("projects/sat-io/open-datasets/GWL_FCS30")
        .sort("system:time_start", False)
        .first()
        .clip(aoi)
    )


def get_nighttime_lights(aoi: ee.Geometry) -> ee.Image:
    """Get nighttime lights data for 2020.

    Args:
        aoi: Area of interest geometry

    Returns:
        ee.Image: Nighttime lights image
    """
    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
    filtered_viirs = viirs.filterDate("2020-01-01", "2021-01-01")
    ntl_composite = filtered_viirs.qualityMosaic("avg_rad").clip(aoi)
    return ntl_composite.toFloat()


def calculate_pixel_vulnerability(
    population_img: ee.Image, nightlights_img: ee.Image
) -> ee.Image:
    """Calculate vulnerability score at pixel level.

    Higher values indicate higher vulnerability. Uses log scaling and
    population mask.

    Args:
        population_img: Population image
        nightlights_img: Nighttime lights image

    Returns:
        ee.Image: Vulnerability image
    """
    # Add small constant to population to avoid division by zero
    pop_mask = population_img.gt(0)
    population_masked = population_img.updateMask(pop_mask)
    safe_population = population_masked.add(0.1)

    # Calculate light per person ratio
    light_per_person = nightlights_img.divide(safe_population)
    log_light_per_person = light_per_person.add(0.001).log()

    # Get percentiles for normalization
    percentiles = log_light_per_person.reduceRegion(
        reducer=ee.Reducer.percentile([2, 98]),
        geometry=aoi,
        scale=500,
        maxPixels=1e9,
    )

    # Extract min and max values
    min_val = ee.Number(percentiles.get("avg_rad_p2"))
    max_val = ee.Number(percentiles.get("avg_rad_p98"))

    # Normalize to 0-1 scale
    normalized_lpp = log_light_per_person.subtract(min_val).divide(
        max_val.subtract(min_val)
    )
    normalized_lpp = normalized_lpp.clamp(0, 1)

    # Invert scale so higher values = higher vulnerability
    vulnerability = ee.Image(1).subtract(normalized_lpp)
    vulnerability = vulnerability.select([0], ["vulnerability"])

    # Apply smoothing
    return vulnerability.focal_mean(radius=500, units="meters")


def start_export_task(
    geotiff: ee.Image,
    description: str,
    bucket: str,
    fileNamePrefix: str,
    scale: Optional[int] = None,
) -> ee.batch.Task:
    """Start an Earth Engine export task to Cloud Storage.

    Args:
        geotiff: The Earth Engine image to export
        description: Description of the export task
        bucket: GCS bucket name
        fileNamePrefix: Path and filename prefix in the bucket
        scale: Optional scale in meters

    Returns:
        ee.batch.Task: The export task
    """
    logger.info(f"Starting export: {description}")
    task = ee.batch.Export.image.toCloudStorage(
        image=geotiff,
        description=description,
        bucket=bucket,
        fileNamePrefix=fileNamePrefix,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    if scale is not None:
        task.config["scale"] = scale
    task.start()
    return task


def monitor_tasks(tasks: List[ee.batch.Task], sleep_interval: int = 10) -> None:
    """Monitor the completion status of Earth Engine tasks.

    Args:
        tasks: List of Earth Engine tasks to monitor
        sleep_interval: Time in seconds between status checks
    """
    logger.info("Monitoring tasks...")
    completed_tasks: Set[str] = set()

    while len(completed_tasks) < len(tasks):
        for task in tasks:
            if task.id in completed_tasks:
                continue

            try:
                status = task.status()
                state = status.get("state")

                if state in ["COMPLETED", "FAILED", "CANCELLED"]:
                    if state == "COMPLETED":
                        logger.info(f"Task {task.id} completed successfully")
                    elif state == "FAILED":
                        error_msg = status.get("error_message", "No error message")
                        logger.error(f"Task {task.id} failed: {error_msg}")
                    elif state == "CANCELLED":
                        logger.warning(f"Task {task.id} was cancelled")
                    completed_tasks.add(task.id)
                else:
                    logger.info(f"Task {task.id} is {state}")
            except ee.EEException as e:
                logger.error(f"Error checking task {task.id}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")

        time.sleep(sleep_interval)

    logger.info("All tasks have been processed")


def export_all_datasets(aoi: ee.Geometry) -> List[ee.batch.Task]:
    """Export all datasets to Cloud Storage.

    Only exports datasets that don't already exist in the bucket.

    Args:
        aoi: Area of interest geometry

    Returns:
        List[ee.batch.Task]: List of export tasks
    """
    export_tasks = []

    # Population export
    population = get_population(aoi)
    pop_prefix = "draft_data/worldpop_2020"
    if not data_exists(BUCKET_NAME, pop_prefix):
        pop_task = start_export_task(
            population, "worldpop_2020", BUCKET_NAME, pop_prefix
        )
        export_tasks.append(pop_task)
    else:
        logger.info(f"Population data already exists at {pop_prefix}")

    # Impervious surface export
    impervious_surface = get_impervious_surface(aoi)
    gisa_prefix = "draft_data/gisa_2019"
    if not data_exists(BUCKET_NAME, gisa_prefix):
        gisa_task = start_export_task(
            impervious_surface, "gisa_2019", BUCKET_NAME, gisa_prefix
        )
        export_tasks.append(gisa_task)
    else:
        logger.info(f"Impervious surface data already exists at {gisa_prefix}")

    # Wetlands export
    wetlands = get_wetlands(aoi)
    wetlands_prefix = "draft_data/wetlands_2022"
    if not data_exists(BUCKET_NAME, wetlands_prefix):
        wetlands_task = start_export_task(
            wetlands, "wetlands_2022", BUCKET_NAME, wetlands_prefix
        )
        export_tasks.append(wetlands_task)
    else:
        logger.info(f"Wetlands data already exists at {wetlands_prefix}")

    # Nighttime lights export
    nighttime_lights = get_nighttime_lights(aoi)
    ntl_prefix = "draft_data/nighttime_lights_2021"
    if not data_exists(BUCKET_NAME, ntl_prefix):
        ntl_task = start_export_task(
            nighttime_lights, "nighttime_lights_2021", BUCKET_NAME, ntl_prefix
        )
        export_tasks.append(ntl_task)
    else:
        logger.info(f"Nighttime lights data already exists at {ntl_prefix}")

    # Vulnerability export
    vulnerability = calculate_pixel_vulnerability(population, nighttime_lights)
    vuln_prefix = "draft_data/vulnerability_2021"
    if not data_exists(BUCKET_NAME, vuln_prefix):
        vuln_task = start_export_task(
            vulnerability, "vulnerability_2021", BUCKET_NAME, vuln_prefix
        )
        export_tasks.append(vuln_task)
    else:
        logger.info(f"Vulnerability data already exists at {vuln_prefix}")

    return export_tasks


def load_exported_datasets() -> Dict[str, ee.Image]:
    """Load all exported datasets from GCS back into Earth Engine images.

    Returns:
        Dict[str, ee.Image]: Dictionary of Earth Engine images keyed by dataset name
    """
    datasets = {}

    dataset_configs = {
        "population": "draft_data/worldpop_2020",
        "impervious": "draft_data/gisa_2019",
        "wetlands": "draft_data/wetlands_2022",
        "nighttime_lights": "draft_data/nighttime_lights_2021",
        "vulnerability": "draft_data/vulnerability_2021",
    }

    for name, prefix in dataset_configs.items():
        logger.info(f"\nLoading {name} dataset:")
        logger.info(f"Prefix: {prefix}")

        file_urls = list_and_check_gcs_files(BUCKET_NAME, prefix)
        logger.info(f"Found files: {file_urls}")

        if file_urls:
            asset_id = file_urls[0]
            logger.info(f"Using asset_id: {asset_id}")
            try:
                image = ee.Image.loadGeoTIFF(asset_id)
                datasets[name] = image
                logger.info(f"Successfully loaded {name} dataset")
            except Exception as e:
                logger.error(f"Error loading {name} dataset: {e}")
                logger.error(f"Error type: {type(e)}")
        else:
            logger.warning(f"No files found for {name} dataset")

    return datasets


if __name__ == "__main__":
    # Get the area of interest
    municipalities = get_municipalities()
    basins_gdf = gpd.read_file(
        "/home/nissim/Documents/dev/ciut-tablero/data/dipsoh_cuencas.geojson"
    )
    basins_ee = geemap.geopandas_to_ee(basins_gdf)
    intersecting_basins = basins_ee.filterBounds(municipalities.geometry())
    aoi = intersecting_basins.geometry()

    # Start all exports and get tasks
    tasks = export_all_datasets(aoi)

    # Monitor all tasks
    monitor_tasks(tasks)
