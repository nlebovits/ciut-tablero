import re
import time

import ee
import geemap
import geopandas as gpd
from google.cloud import storage

ee.Authenticate()

ee.Initialize(project="ee-ciut")

# Constants
BUCKET_NAME = "ciut-tablero"
GOOGLE_CLOUD_PROJECT = "ee-ciut"


def data_exists(bucket_name, prefix):
    storage_client = storage.Client(project=GOOGLE_CLOUD_PROJECT)
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    return len(blobs) > 0


def list_and_check_gcs_files(bucket_name, prefix):
    """Check if files exist in a GCS bucket folder and list them if they do."""
    # Create a GCS client
    client = storage.Client()

    # Obtain the bucket object
    bucket = client.bucket(bucket_name)

    # List blobs with the specified prefix
    blobs = list(bucket.list_blobs(prefix=prefix))

    # Check if any files exist with the specified prefix
    if len(blobs) == 0:
        print(f"No files found with prefix '{prefix}' in bucket '{bucket_name}'.")
        return []

    # List and return all files with the specified prefix
    file_urls = [
        f"gs://{bucket_name}/{blob.name}"
        for blob in blobs
        if blob.name.endswith(".tif")
    ]
    return file_urls


def extract_date_from_filename(filename):
    # Use a regular expression to find dates in the format YYYY-MM-DD
    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
    if match:
        return match.group(0)  # Return the first match
    else:
        return None


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


def get_nighttime_lights(aoi):
    # Nighttime lights with fixed year (2020)
    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
    # Fixed date range for 2020
    start_date = "2020-01-01"
    end_date = "2021-01-01"
    filtered_viirs = viirs.filterDate(start_date, end_date)
    ntl_composite = filtered_viirs.qualityMosaic("avg_rad").clip(aoi)

    # Convert all bands to float32 in one step
    ntl_composite_float32 = ntl_composite.toFloat()

    return ntl_composite_float32


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


def start_export_task(geotiff, description, bucket, fileNamePrefix, scale=None):
    """
    Start an Earth Engine export task to Cloud Storage.

    Args:
        geotiff: The Earth Engine image to export
        description: Description of the export task
        bucket: GCS bucket name
        fileNamePrefix: Path and filename prefix in the bucket
        scale: Optional scale in meters. If None, uses image default.
    """
    print(f"Starting export: {description}")
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


def monitor_tasks(tasks, sleep_interval=10):
    """
    Monitors the completion status of provided Earth Engine tasks.

    Parameters:
    - tasks: A list of Earth Engine tasks to monitor.
    - sleep_interval: Time in seconds to wait between status checks (default is 10 seconds).
    """
    print("Monitoring tasks...")
    completed_tasks = set()
    while len(completed_tasks) < len(tasks):
        for task in tasks:
            if task.id in completed_tasks:
                continue

            try:
                status = task.status()
                state = status.get("state")

                if state in ["COMPLETED", "FAILED", "CANCELLED"]:
                    if state == "COMPLETED":
                        print(f"Task {task.id} completed successfully.")
                    elif state == "FAILED":
                        print(
                            f"Task {task.id} failed with error: {status.get('error_message', 'No error message provided.')}"
                        )
                    elif state == "CANCELLED":
                        print(f"Task {task.id} was cancelled.")
                    completed_tasks.add(task.id)
                else:
                    print(f"Task {task.id} is {state}.")
            except ee.EEException as e:
                print(f"Error checking status of task {task.id}: {e}. Will retry...")
            except Exception as general_error:
                print(f"Unexpected error: {general_error}. Will retry...")

        # Wait before the next status check to limit API requests and give time for tasks to progress
        time.sleep(sleep_interval)

    print("All tasks have been processed.")


def export_all_datasets(aoi):
    """
    Export all datasets to Cloud Storage and return a list of tasks to monitor.
    Only exports datasets that don't already exist in the bucket.
    """
    # List to store all export tasks
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
        print(f"Population data already exists at {pop_prefix}")

    # Impervious surface export
    impervious_surface = get_impervious_surface(aoi)
    gisa_prefix = "draft_data/gisa_2019"
    if not data_exists(BUCKET_NAME, gisa_prefix):
        gisa_task = start_export_task(
            impervious_surface, "gisa_2019", BUCKET_NAME, gisa_prefix
        )
        export_tasks.append(gisa_task)
    else:
        print(f"Impervious surface data already exists at {gisa_prefix}")

    # Wetlands export
    wetlands = get_wetlands(aoi)
    wetlands_prefix = "draft_data/wetlands_2022"
    if not data_exists(BUCKET_NAME, wetlands_prefix):
        wetlands_task = start_export_task(
            wetlands, "wetlands_2022", BUCKET_NAME, wetlands_prefix
        )
        export_tasks.append(wetlands_task)
    else:
        print(f"Wetlands data already exists at {wetlands_prefix}")

    # Nighttime lights export
    nighttime_lights = get_nighttime_lights(aoi)
    ntl_prefix = "draft_data/nighttime_lights_2021"
    if not data_exists(BUCKET_NAME, ntl_prefix):
        ntl_task = start_export_task(
            nighttime_lights, "nighttime_lights_2021", BUCKET_NAME, ntl_prefix
        )
        export_tasks.append(ntl_task)
    else:
        print(f"Nighttime lights data already exists at {ntl_prefix}")

    # Vulnerability export
    vulnerability = calculate_pixel_vulnerability(population, nighttime_lights)
    vuln_prefix = "draft_data/vulnerability_2021"
    if not data_exists(BUCKET_NAME, vuln_prefix):
        vuln_task = start_export_task(
            vulnerability, "vulnerability_2021", BUCKET_NAME, vuln_prefix
        )
        export_tasks.append(vuln_task)
    else:
        print(f"Vulnerability data already exists at {vuln_prefix}")

    return export_tasks


# Main execution
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
