import os
import tempfile
import time

import ee
import geemap
import rasterio

from utils.auth import get_bucket_name, get_credentials, get_storage_client


def get_bucket():
    """Get the GCP bucket instance"""
    credentials = get_credentials()
    client = get_storage_client(credentials)
    bucket_name = get_bucket_name()
    return client.bucket(bucket_name)


def blob_exists(blob_name):
    """Check if a blob exists in the bucket"""
    bucket = get_bucket()
    blob = bucket.blob(blob_name)
    return blob.exists()


def download_blob(blob_name):
    """Download a blob from the bucket to a temporary file"""
    bucket = get_bucket()
    blob = bucket.blob(blob_name)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_file:
        blob.download_to_filename(temp_file.name)
        return temp_file.name


def export_to_gcs(image, blob_name, aoi):
    """Export an Earth Engine image directly to GCS as a COG"""

    # Create the export task
    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=f"Export_{blob_name}",
        bucket=get_bucket_name(),
        fileNamePrefix=blob_name.replace(".tif", ""),
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )

    # Start the task
    task.start()

    # Wait for the task to complete
    while task.status()["state"] in ["READY", "RUNNING"]:
        time.sleep(5)

    if task.status()["state"] != "COMPLETED":
        raise Exception(f"Export failed: {task.status()}")


def load_cached_layer(blob_name, aoi, ee_loader_func):
    """Load a layer from cache or GEE, with caching"""
    # Check if the layer exists in the bucket
    if blob_exists(blob_name):
        # Download and load the COG
        temp_file = download_blob(blob_name)
        try:
            with rasterio.open(temp_file):
                # Convert to EE image
                image = geemap.raster_to_ee(temp_file)
                return image
        finally:
            # Clean up the temporary file
            os.unlink(temp_file)
    else:
        # Load from GEE and cache
        image = ee_loader_func(aoi)
        export_to_gcs(image, blob_name, aoi)
        return image
