import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import ee
import geemap
import geopandas as gpd
import pandas as pd
from etl import get_municipalities

sys.path.append(os.path.abspath(".."))  # Add the parent directory to Python path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("timeseries_etl.log")],
)
logger = logging.getLogger(__name__)

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project="ee-ciut")

# Constants
REQUEST_DELAY = 1.0  # Delay between Earth Engine requests in seconds


def safe_ee_request(func, *args, **kwargs):
    """
    Safely execute an Earth Engine request with retries and delays.

    Args:
        func: Function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function call
    """
    try:
        result = func(*args, **kwargs)
        time.sleep(REQUEST_DELAY)  # Add delay between requests
        return result
    except Exception as e:
        logger.error(f"Error in Earth Engine request: {str(e)}")
        raise


def initialize_basins() -> Tuple[ee.FeatureCollection, ee.Geometry]:
    """
    Initialize and prepare basin data for analysis.

    Returns:
        Tuple[ee.FeatureCollection, ee.Geometry]: A tuple containing:
            - intersecting_basins: Earth Engine FeatureCollection of basins that intersect with municipalities
            - aoi: Earth Engine Geometry representing the area of interest
    """
    logger.info("Initializing basins data")
    municipalities = get_municipalities()
    basins_gdf = gpd.read_file(
        "/home/nissim/Documents/dev/ciut-tablero/data/dipsoh_cuencas.geojson"
    )

    # Create a unique name by combining nombre and sucuenca_n with comma separation
    basins_gdf["unique_name"] = basins_gdf.apply(
        lambda row: f"{row['nombre']}, {row['sucuenca_n']}"
        if pd.notnull(row.get("sucuenca_n")) and row["sucuenca_n"] != ""
        else row["nombre"],
        axis=1,
    )

    # Convert to Earth Engine format
    basins_ee = geemap.geopandas_to_ee(basins_gdf)
    intersecting_basins = basins_ee.filterBounds(municipalities.geometry())
    aoi = intersecting_basins.geometry()

    logger.info(f"Initialized {len(intersecting_basins.getInfo()['features'])} basins")
    return intersecting_basins, aoi


def calculate_rolling_averages(
    df: pd.DataFrame, value_col: str, window: int = 3
) -> pd.DataFrame:
    """
    Calculate rolling averages for a given value column.

    Args:
        df: Input DataFrame containing the data
        value_col: Name of the column to calculate rolling averages for
        window: Size of the rolling window (default: 3)

    Returns:
        pd.DataFrame: DataFrame with added rolling average column
    """
    logger.info(
        f"Calculating rolling averages for {value_col} with window size {window}"
    )
    df_result = df.copy()
    rolling_col = f"rolling_avg_{value_col}"
    df_result[rolling_col] = float("nan")

    for basin in df_result["unique_name"].unique():
        basin_data = df_result[df_result["unique_name"] == basin].sort_values("year")

        if len(basin_data) >= window:
            basin_data.loc[:, rolling_col] = (
                basin_data[value_col]
                .rolling(window=window, center=True, min_periods=1)
                .mean()
            )
            df_result.loc[basin_data.index, rolling_col] = basin_data[rolling_col]
        else:
            avg_value = basin_data[value_col].mean()
            df_result.loc[df_result["unique_name"] == basin, rolling_col] = avg_value

    return df_result


def process_population_year(
    year: int, intersecting_basins: ee.FeatureCollection
) -> List[Dict[str, Any]]:
    """
    Process population data for a single year.
    """
    logger.info(f"Processing population data for year {year}")

    try:
        population_collection = ee.ImageCollection("WorldPop/GP/100m/pop").filter(
            ee.Filter.stringContains("system:index", "ARG_")
        )

        def extract_year(img: ee.Image) -> ee.Image:
            year = ee.Number.parse(ee.String(img.get("system:index")).slice(4, 8))
            return img.set("year", year)

        population_with_years = population_collection.map(extract_year)
        pop_img = population_with_years.filter(ee.Filter.eq("year", year)).first()

        if pop_img is None:
            logger.warning(f"No population data found for year {year}")
            return []

        def calculate_pop(basin: ee.Feature) -> ee.Feature:
            pop_sum = pop_img.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=basin.geometry(),
                scale=100,
                maxPixels=1e10,
            ).get("population")
            return basin.set({"year": year, "population": pop_sum})

        basins_with_pop = intersecting_basins.map(calculate_pop)
        result = safe_ee_request(basins_with_pop.getInfo)

        basin_data = []
        for feature in result["features"]:
            props = feature["properties"]
            basin_data.append(
                {
                    "unique_name": props.get("unique_name", ""),
                    "year": year,
                    "population": props.get("population", 0),
                }
            )

        logger.info(
            f"Processed population data for {len(basin_data)} basins in year {year}"
        )
        return basin_data
    except Exception as e:
        logger.error(f"Error processing population data for year {year}: {str(e)}")
        return []


def get_population_data(
    intersecting_basins: ee.FeatureCollection, years: List[int]
) -> pd.DataFrame:
    """
    Extract population data for basins over time.
    """
    logger.info(f"Starting population data extraction for years {years}")
    start_time = time.time()

    all_basin_data = []
    for year in years:
        basin_data = process_population_year(year, intersecting_basins)
        all_basin_data.extend(basin_data)

    df = pd.DataFrame(all_basin_data)
    logger.info(
        f"Population data extraction completed in {time.time() - start_time:.2f} seconds"
    )
    return calculate_rolling_averages(df, "population")


def process_impervious_year(
    year: int, intersecting_basins: ee.FeatureCollection, aoi: ee.Geometry
) -> List[Dict[str, Any]]:
    """
    Process impervious data for a single year.
    """
    logger.info(f"Processing impervious data for year {year}")

    try:
        gisa = (
            ee.ImageCollection("projects/sat-io/open-datasets/GISA_1972_2019")
            .mosaic()
            .clip(aoi)
        )

        if year >= 2000:
            year_index = year - 2000
            max_value = year_index + 1
            year_mask = gisa.gt(0).And(gisa.lte(max_value))

            basin_areas = intersecting_basins.map(
                lambda basin: basin.set("area_m2", basin.geometry().area(maxError=1))
            )

            impervious_area = year_mask.multiply(ee.Image.pixelArea())

            zonal_stats = impervious_area.reduceRegions(
                collection=basin_areas,
                reducer=ee.Reducer.sum(),
                scale=30,
            )

            results = safe_ee_request(zonal_stats.getInfo)

            basin_data = []
            for feature in results["features"]:
                unique_name = feature["properties"]["unique_name"]
                basin_area = feature["properties"]["area_m2"]
                impervious_area = feature["properties"].get("sum", 0) or 0
                impervious_pct = (impervious_area / basin_area) * 100

                basin_data.append(
                    {
                        "unique_name": unique_name,
                        "year": year,
                        "impervious_pct": impervious_pct,
                    }
                )

            logger.info(
                f"Processed impervious data for {len(basin_data)} basins in year {year}"
            )
            return basin_data

        logger.warning(f"Skipping impervious data for year {year} (before 2000)")
        return []
    except Exception as e:
        logger.error(f"Error processing impervious data for year {year}: {str(e)}")
        return []


def get_impervious_data(
    intersecting_basins: ee.FeatureCollection, aoi: ee.Geometry, years: List[int]
) -> pd.DataFrame:
    """
    Extract impervious surface data for basins over time.
    """
    logger.info(f"Starting impervious data extraction for years {years}")
    start_time = time.time()

    all_basin_data = []
    for year in years:
        basin_data = process_impervious_year(year, intersecting_basins, aoi)
        all_basin_data.extend(basin_data)

    df = pd.DataFrame(all_basin_data)
    logger.info(
        f"Impervious data extraction completed in {time.time() - start_time:.2f} seconds"
    )
    return calculate_rolling_averages(df, "impervious_pct")


def process_wetlands_year(
    year: int, intersecting_basins: ee.FeatureCollection, aoi: ee.Geometry
) -> List[Dict[str, Any]]:
    """
    Process wetlands data for a single year.
    """
    logger.info(f"Processing wetlands data for year {year}")

    try:
        wetlands_collection = ee.ImageCollection(
            "projects/sat-io/open-datasets/GWL_FCS30"
        )
        wetland_codes = [180, 181, 182, 183, 186, 187]

        def create_wetland_mask(image: ee.Image) -> ee.Image:
            mask = image.eq(wetland_codes[0])
            for code in wetland_codes[1:]:
                mask = mask.Or(image.eq(code))
            return mask

        year_img = wetlands_collection.filter(
            ee.Filter.eq("system:index", f"GWL_FCS30_{year}")
        ).first()

        if year_img is None:
            logger.warning(f"No wetlands data found for year {year}")
            return []

        year_img = year_img.clip(aoi)
        wetland_mask = create_wetland_mask(year_img)
        wetland_area = wetland_mask.multiply(ee.Image.pixelArea())

        basin_areas = intersecting_basins.map(
            lambda basin: basin.set("area_m2", basin.geometry().area(maxError=1))
        )

        zonal_stats = wetland_area.reduceRegions(
            collection=basin_areas,
            reducer=ee.Reducer.sum(),
            scale=30,
        )

        results = safe_ee_request(zonal_stats.getInfo)

        basin_data = []
        for feature in results["features"]:
            props = feature["properties"]
            unique_name = props.get("unique_name", "")
            basin_area = props.get("area_m2", 0)
            wetland_area_m2 = props.get("sum", 0) or 0

            if basin_area > 0:
                wetland_pct = min((wetland_area_m2 / basin_area) * 100, 100)
            else:
                wetland_pct = 0

            basin_data.append(
                {
                    "unique_name": unique_name,
                    "year": year,
                    "wetland_pct": wetland_pct,
                }
            )

        logger.info(
            f"Processed wetlands data for {len(basin_data)} basins in year {year}"
        )
        return basin_data
    except Exception as e:
        logger.error(f"Error processing wetlands data for year {year}: {str(e)}")
        return []


def get_wetlands_data(
    intersecting_basins: ee.FeatureCollection, aoi: ee.Geometry, years: List[int]
) -> pd.DataFrame:
    """
    Extract wetlands data for basins over time.
    """
    logger.info(f"Starting wetlands data extraction for years {years}")
    start_time = time.time()

    all_basin_data = []
    for year in years:
        basin_data = process_wetlands_year(year, intersecting_basins, aoi)
        all_basin_data.extend(basin_data)

    df = pd.DataFrame(all_basin_data)
    logger.info(
        f"Wetlands data extraction completed in {time.time() - start_time:.2f} seconds"
    )
    return calculate_rolling_averages(df, "wetland_pct")


def process_vulnerability_year(
    year: int, intersecting_basins: ee.FeatureCollection, aoi: ee.Geometry
) -> List[Dict[str, Any]]:
    """
    Process vulnerability data for a single year.
    """
    logger.info(f"Processing vulnerability data for year {year}")

    try:

        def calculate_pixel_vulnerability(
            population_img: ee.Image, nightlights_img: ee.Image
        ) -> ee.Image:
            pop_mask = population_img.gt(0)
            population_masked = population_img.updateMask(pop_mask)
            safe_population = population_masked.add(0.1)
            light_per_person = nightlights_img.divide(safe_population)
            log_light_per_person = light_per_person.add(0.001).log()

            percentiles = log_light_per_person.reduceRegion(
                reducer=ee.Reducer.percentile([2, 98]), geometry=aoi, scale=500
            )

            min_val = ee.Number(percentiles.get("avg_rad_p2"))
            max_val = ee.Number(percentiles.get("avg_rad_p98"))
            normalized_lpp = log_light_per_person.subtract(min_val).divide(
                max_val.subtract(min_val)
            )
            normalized_lpp = normalized_lpp.clamp(0, 1)
            vulnerability = ee.Image(1).subtract(normalized_lpp)
            return vulnerability.select([0], ["vulnerability"])

        population = (
            ee.ImageCollection("WorldPop/GP/100m/pop")
            .filter(ee.Filter.eq("system:index", f"ARG_{year}"))
            .first()
            .clip(aoi)
        )

        nightlights = (
            ee.ImageCollection("projects/sat-io/open-datasets/srunet-npp-viirs-ntl")
            .map(
                lambda img: img.set(
                    "year", ee.Number.parse(ee.String(img.get("id_no")).slice(-4))
                )
            )
            .filter(ee.Filter.eq("year", year))
            .first()
            .select("b1")
            .rename("avg_rad")
            .clip(aoi)
        )

        vulnerability = calculate_pixel_vulnerability(population, nightlights)
        vulnerability_smoothed = vulnerability.focal_mean(radius=500, units="meters")

        basin_areas = intersecting_basins.map(
            lambda basin: basin.set("area_m2", basin.geometry().area())
        )

        mean_vulnerability = vulnerability_smoothed.reduceRegions(
            collection=basin_areas, reducer=ee.Reducer.mean(), scale=100
        )

        population_sum = population.reduceRegions(
            collection=basin_areas, reducer=ee.Reducer.sum(), scale=100
        )

        mean_results = safe_ee_request(mean_vulnerability.getInfo)
        pop_results = safe_ee_request(population_sum.getInfo)

        pop_by_basin = {
            feature["properties"].get("unique_name", ""): feature["properties"].get(
                "sum", 0
            )
            for feature in pop_results["features"]
        }

        basin_data = []
        for feature in mean_results["features"]:
            unique_name = feature["properties"].get("unique_name", "")
            mean_vuln = feature["properties"].get("mean", 0) or 0
            basin_pop = pop_by_basin.get(unique_name, 0) or 1

            vulnerability_score = mean_vuln * 100
            basin_data.append(
                {
                    "unique_name": unique_name,
                    "year": year,
                    "vulnerability_score": vulnerability_score,
                    "population": basin_pop,
                }
            )

        logger.info(
            f"Processed vulnerability data for {len(basin_data)} basins in year {year}"
        )
        return basin_data
    except Exception as e:
        logger.error(f"Error processing vulnerability data for year {year}: {str(e)}")
        return []


def get_vulnerability_data(
    intersecting_basins: ee.FeatureCollection, aoi: ee.Geometry, years: List[int]
) -> pd.DataFrame:
    """
    Calculate vulnerability scores for basins over time.
    """
    logger.info(f"Starting vulnerability data extraction for years {years}")
    start_time = time.time()

    all_basin_data = []
    for year in years:
        basin_data = process_vulnerability_year(year, intersecting_basins, aoi)
        all_basin_data.extend(basin_data)

    df = pd.DataFrame(all_basin_data)
    logger.info(
        f"Vulnerability data extraction completed in {time.time() - start_time:.2f} seconds"
    )
    return calculate_rolling_averages(df, "vulnerability_score")


def combine_all_data(
    pop_df: pd.DataFrame,
    imp_df: pd.DataFrame,
    wet_df: pd.DataFrame,
    vul_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine all dataframes into a single tidy dataframe.

    Args:
        pop_df: DataFrame containing population data
        imp_df: DataFrame containing impervious surface data
        wet_df: DataFrame containing wetlands data
        vul_df: DataFrame containing vulnerability data

    Returns:
        pd.DataFrame: Combined tidy dataframe with all metrics
    """
    logger.info("Combining all data into a single tidy dataframe")
    start_time = time.time()

    # Create a complete index of all basin-year combinations
    all_basins = (
        set(pop_df["unique_name"].unique())
        | set(imp_df["unique_name"].unique())
        | set(wet_df["unique_name"].unique())
        | set(vul_df["unique_name"].unique())
    )
    all_years = (
        set(pop_df["year"].unique())
        | set(imp_df["year"].unique())
        | set(wet_df["year"].unique())
        | set(vul_df["year"].unique())
    )

    # Create a MultiIndex with all combinations
    index = pd.MultiIndex.from_product(
        [sorted(all_basins), sorted(all_years)], names=["unique_name", "year"]
    )

    # Create an empty dataframe with the complete index
    combined_df = pd.DataFrame(index=index).reset_index()

    # Rename overlapping columns before merging
    pop_df = pop_df.rename(columns={"population": "population_from_pop"})
    vul_df = vul_df.rename(columns={"population": "population_from_vul"})

    # Merge each dataframe, keeping all rows
    for df in [pop_df, imp_df, wet_df, vul_df]:
        combined_df = pd.merge(combined_df, df, on=["unique_name", "year"], how="left")

    logger.info(f"Data combination completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Final dataframe shape: {combined_df.shape}")
    return combined_df


def process_data_extraction(
    func_name: str, basins: ee.FeatureCollection, aoi: ee.Geometry, years: List[int]
) -> pd.DataFrame:
    """
    Process data extraction for a given function name.

    Args:
        func_name: Name of the function to call
        basins: Earth Engine FeatureCollection of basins
        aoi: Earth Engine Geometry representing the area of interest
        years: List of years to process

    Returns:
        pd.DataFrame: Extracted data
    """
    # Initialize Earth Engine in the new process
    ee.Initialize(project="ee-ciut")

    func_map = {
        "population": get_population_data,
        "impervious": get_impervious_data,
        "wetlands": get_wetlands_data,
        "vulnerability": get_vulnerability_data,
    }

    func = func_map[func_name]
    if func_name == "population":
        return func(basins, years)
    return func(basins, aoi, years)


def main() -> pd.DataFrame:
    """
    Main function to run the ETL process.
    """
    logger.info("Starting ETL process")
    start_time = time.time()

    try:
        # Initialize basins and area of interest
        intersecting_basins, aoi = initialize_basins()

        # Define years for analysis
        years = list(range(2000, 2023))  # 2000 to 2022
        logger.info(f"Processing data for years: {years}")

        # Process each metric sequentially
        logger.info("Processing population data")
        pop_df = get_population_data(intersecting_basins, years)

        logger.info("Processing impervious surface data")
        imp_df = get_impervious_data(intersecting_basins, aoi, years)

        logger.info("Processing wetlands data")
        wet_df = get_wetlands_data(intersecting_basins, aoi, years)

        logger.info("Processing vulnerability data")
        vul_df = get_vulnerability_data(intersecting_basins, aoi, years)

        # Combine all data into a single tidy dataframe
        final_df = combine_all_data(pop_df, imp_df, wet_df, vul_df)

        # Write the final dataframe to parquet
        output_path = Path("data/timeseries_data.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Writing final dataframe to {output_path}")
        final_df.to_parquet(output_path, index=False)
        logger.info(f"Successfully wrote {len(final_df)} rows to {output_path}")

        logger.info(f"ETL process completed in {time.time() - start_time:.2f} seconds")
        return final_df
    except Exception as e:
        logger.error(f"Error in ETL process: {str(e)}")
        raise


if __name__ == "__main__":
    final_df = main()
