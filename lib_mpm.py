from geopandas import GeoSeries
import geopandas as gpd
import numpy as np
import os
from packaging import version
import pandas as pd
from pandas import Series
from pyproj import CRS, Geod
import rasterio
from rasterio.windows import Window
from shapely import __version__ as shapely_version
from shapely.geometry import Point, LineString, MultiLineString, GeometryCollection, MultiPoint
from shapely.ops import nearest_points
from shapely.strtree import STRtree
from skimage.feature import graycomatrix, graycoprops
from typing import List, Literal, Tuple, Union

# -------------------------------------

def get_ellipsoid_name_gpd(crs):
    """
    Helper function to get the correct ellipsoid name for Geod
    
    Ehsan Farahbakhsh
    EarthByte Group, School of Geosciences, The University of Sydney, Sydney, Australia
    Date: 07/09/2024
    """
    ellipsoid_name = crs.ellipsoid.name.lower()
    if 'grs' in ellipsoid_name and '1980' in ellipsoid_name:
        return 'GRS80'
    elif 'wgs' in ellipsoid_name and '84' in ellipsoid_name:
        return 'WGS84'
    else:
        return 'WGS84'  # Default to WGS84 if unknown

def get_nearest_point(point, geometry):
    """
    Helper function to get the nearest point on a geometry
    
    Ehsan Farahbakhsh
    EarthByte Group, School of Geosciences, The University of Sydney, Sydney, Australia
    Date: 17/09/2024
    """
    return nearest_points(point, geometry)[1]

def extract_point(geom):
    """
    Extract a point from a geometry, handling all geometry types
    
    Ehsan Farahbakhsh
    EarthByte Group, School of Geosciences, The University of Sydney, Sydney, Australia
    Date: 17/09/2024
    """
    if isinstance(geom, Point):
        return geom
    elif isinstance(geom, (LineString, MultiLineString)):
        return Point(geom.interpolate(0, normalized=True))
    elif isinstance(geom, GeometryCollection):
        for part in geom.geoms:
            if not part.is_empty:
                return extract_point(part)
    elif isinstance(geom, MultiPoint):
        return geom[0]
    # If we can't extract a specific point, return the centroid
    return geom.centroid

def get_suitable_projected_crs(gdf):
    """
    Helper function to get a suitable projected CRS based on the data's extent
    
    Ehsan Farahbakhsh
    EarthByte Group, School of Geosciences, The University of Sydney, Sydney, Australia
    Date: 07/09/2024
    """
    bounds = gdf.total_bounds
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2
    utm_zone = int((center_lon + 180) / 6) + 1
    epsg = 32600 + utm_zone + (0 if center_lat >= 0 else 100)
    return CRS(f"EPSG:{epsg}")

def get_dist_line(
    xs: Union[List[float], np.ndarray, Series],
    ys: Union[List[float], np.ndarray, Series],
    line_files: List[str],
    distance_type: Literal['euclidean', 'geodesic'] = 'euclidean',
    input_crs: str = 'EPSG:4283'
) -> pd.DataFrame:
    """
    Calculate the distance from points to linear features.

    Args:
        xs (Union[List[float], np.ndarray, Series]): List, array, or Series of x-coordinates of the points.
        ys (Union[List[float], np.ndarray, Series]): List, array, or Series of y-coordinates of the points.
        line_files (List[str]): List of file paths to linear features.
        distance_type (Literal['euclidean', 'geodesic'], optional): Type of distance calculation.
                                                                    Defaults to 'euclidean'.
        input_crs (str, optional): Coordinate Reference System of input data. Defaults to 'EPSG:4283'.

    Returns:
        pd.DataFrame: A DataFrame where each column represents distances to a line in meters.

    Raises:
        ValueError: If input parameters are invalid.
        FileNotFoundError: If a line file is not found.
        gpd.io.file.DriverError: If there's an error reading a line file.

    Notes:
        - The function supports both Euclidean and geodesic distance calculations.
        - For geodesic calculations, the function uses the ellipsoid associated with the input CRS.
        - If the input CRS is projected and geodesic distance is requested, the data is converted to a geographic CRS.
        - If the input CRS is geographic and Euclidean distance is requested, the data is projected to a suitable local projection.
        
        Ehsan Farahbakhsh
        EarthByte Group, School of Geosciences, The University of Sydney, Sydney, Australia
        Date: 17/09/2024
    """
    # Input validation
    if not isinstance(xs, (list, np.ndarray, Series)) or not isinstance(ys, (list, np.ndarray, Series)):
        raise ValueError("xs and ys must be either lists, numpy arrays, or pandas Series.")
    
    # Convert inputs to numpy arrays
    xs = np.array(xs)
    ys = np.array(ys)
    
    if xs.size != ys.size:
        raise ValueError("xs and ys must have the same length.")
    if not line_files:
        raise ValueError("line_files cannot be empty.")
    if distance_type not in ['euclidean', 'geodesic']:
        raise ValueError("distance_type must be either 'euclidean' or 'geodesic'.")
    if not isinstance(input_crs, str):
        raise ValueError("input_crs must be a string.")

    lines = []
    column_names = []
    
    input_crs = CRS(input_crs)
    
    for file in line_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Line file not found: {file}")
        try:
            gdf = gpd.read_file(file)
            if gdf.crs is None:
                gdf.set_crs(input_crs, inplace=True)
            lines.append(gdf)
            column_names.append(os.path.splitext(os.path.basename(file))[0])
        except gpd.io.file.DriverError as e:
            raise gpd.io.file.DriverError(f"Error reading line file {file}: {str(e)}")
    
    points = [Point(x, y) for x, y in zip(xs, ys)]
    points_gdf = gpd.GeoDataFrame(geometry=points, crs=input_crs)
    
    dist_to_lines = np.zeros((len(points), len(lines)))
    
    is_projected = input_crs.is_projected
    
    if distance_type == 'euclidean':
        if not is_projected:
            # Project to a suitable local projection
            proj_crs = get_suitable_projected_crs(points_gdf)
            points_gdf = points_gdf.to_crs(proj_crs)
            lines = [line.to_crs(proj_crs) for line in lines]
    else:  # geodesic
        if is_projected:
            # Convert to a geographic CRS for geodesic calculations
            geo_crs = CRS("EPSG:4326")
            points_gdf = points_gdf.to_crs(geo_crs)
            lines = [line.to_crs(geo_crs) for line in lines]
        ellps = get_ellipsoid_name_gpd(points_gdf.crs)
        geod = Geod(ellps=ellps)


    shapely_20 = version.parse(shapely_version) >= version.parse("2.0")

    for i, line in enumerate(lines):
        line_tree = STRtree(line.geometry)
        
        if shapely_20:
            nearest_indices = line_tree.nearest(points_gdf.geometry)
            nearest_geoms = GeoSeries(line.geometry.iloc[nearest_indices].values, crs=line.crs)
        else:
            nearest_geoms = GeoSeries([line_tree.nearest(p) for p in points_gdf.geometry], crs=line.crs)
        
        if distance_type == 'euclidean':
            dist_to_lines[:, i] = points_gdf.geometry.distance(nearest_geoms)
        else:  # geodesic
            for j, (p, nearest) in enumerate(zip(points_gdf.geometry, nearest_geoms)):
                p_point = extract_point(p)
                nearest_point = get_nearest_point(p_point, nearest)
                
                _, _, distance = geod.inv(p_point.x, p_point.y, nearest_point.x, nearest_point.y)
                dist_to_lines[j, i] = distance
    
    return pd.DataFrame(dist_to_lines, columns=column_names)

# ----------------------------------------------------------

def get_cat_data(
    xs: Union[List[float], np.ndarray, Series],
    ys: Union[List[float], np.ndarray, Series],
    polygon_files: Union[str, List[str]],
    field: str,
    input_crs: str = 'EPSG:4326'
) -> pd.DataFrame:
    """
    Extract categorical data from geographic polygon files based on given point coordinates.

    This function performs a spatial join between input points and polygons from specified files,
    extracting a given field value for each point based on the polygon it falls within.

    Args:
        xs (Union[List[float], np.ndarray, Series]): List, array, or Series of x-coordinates.
        ys (Union[List[float], np.ndarray, Series]): List, array, or Series of y-coordinates.
        polygon_files (Union[str, List[str]]): Single file path or list of file paths to polygon shapefiles.
        field (str): Name of the field in the polygon files to extract.
        input_crs (str, optional): Coordinate Reference System of input data. Defaults to 'EPSG:4326'.

    Returns:
        pd.DataFrame: DataFrame containing the extracted categorical data for each point and each polygon file.
                      Columns are named after the input files, rows correspond to input points.

    Raises:
        ValueError: If input parameters are invalid.
        FileNotFoundError: If any of the specified polygon files does not exist.
        gpd.io.file.DriverError: If there's an error reading a polygon file.

    Notes:
        - Points that do not fall within any polygon will have 'Null' as their value.
        - If a point falls within multiple polygons, the value from the first polygon is used.
        - The function assumes that all input data are in the same coordinate reference system.
        - Any errors during file processing will be logged, and the file will be skipped.
        
        Ehsan Farahbakhsh
        EarthByte Group, School of Geosciences, The University of Sydney, Sydney, Australia
        Date: 07/09/2024
    """
    # Input validation
    if not isinstance(xs, (list, np.ndarray, Series)) or not isinstance(ys, (list, np.ndarray, Series)):
        raise ValueError("xs and ys must be either lists, numpy arrays, or pandas Series.")
    
    # Convert inputs to numpy arrays
    xs = np.array(xs)
    ys = np.array(ys)
    
    if xs.size != ys.size:
        raise ValueError("Input coordinates must be of equal length.")
    if isinstance(polygon_files, str):
        polygon_files = [polygon_files]
    if not polygon_files:
        raise ValueError("At least one polygon file must be specified.")
    if not isinstance(field, str):
        raise ValueError("field must be a string.")
    if not isinstance(input_crs, str):
        raise ValueError("input_crs must be a string.")

    for file in polygon_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Polygon file not found: {file}")

    # Create GeoDataFrame from input points
    points_gdf = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs=input_crs,
        index=range(xs.size)
    )

    cat_code = pd.DataFrame()
    for file in polygon_files:
        column_name = os.path.splitext(os.path.basename(file))[0]
        try:
            polygon_gdf = gpd.read_file(file)
            
            # Ensure the field exists in the GeoDataFrame
            if field not in polygon_gdf.columns:
                print(f"Warning: Field '{field}' not found in {column_name}. Skipping.")
                continue
            
            # Ensure the CRS matches
            if polygon_gdf.crs != points_gdf.crs:
                polygon_gdf = polygon_gdf.to_crs(points_gdf.crs)
            
            # Spatial join
            joined = gpd.sjoin(points_gdf, polygon_gdf, how="left", predicate="within")
            
            # Group by the original index and aggregate, keeping the first non-null value
            grouped = joined.groupby(joined.index)[field].first()
            cat_code[column_name] = grouped.fillna('Null')
                        
        except gpd.io.file.DriverError as e:
            print(f"Error reading polygon file {file}: {str(e)}")
        except Exception as e:
            print(f"Error processing {column_name}: {str(e)}")
    
    return cat_code.reset_index(drop=True)

# -----------------

def get_grid_stat_features(
    xs: Union[List[float], np.ndarray, Series],
    ys: Union[List[float], np.ndarray, Series],
    raster_paths: Union[str, List[str]],
    buffer_shape: str = 'circle',
    buffer_size: int = 1,
    stats: List[str] = ['mean', 'std', 'min', 'max', 'median'],
    export_pixels: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Calculate statistics for pixels surrounding each point in each raster file.
    Optionally export pixel values for each point.
    
    Args:
        xs (Union[List[float], np.ndarray, Series]): List, array, or Series of x-coordinates.
        ys (Union[List[float], np.ndarray, Series]): List, array, or Series of y-coordinates.
        raster_paths (Union[str, List[str]]): Path to a single raster file or list of paths to georeferenced raster files.
        buffer_shape (str, optional): Shape of the buffer zone ('circle' or 'square'). Defaults to 'circle'.
        buffer_size (int, optional): For 'circle', radius in pixels. For 'square', half of the side length in pixels. Defaults to 1.
        stats (List[str], optional): List of statistics to calculate. 
                                     Supported stats: 'mean', 'std', 'min', 'max', 'median'.
                                     Defaults to ['mean', 'std', 'min', 'max', 'median'].
        export_pixels (bool, optional): If True, export pixel values for each point. Defaults to False.
    
    Returns:
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]: 
            If export_pixels is False, returns a DataFrame with statistics for each point across all raster files.
            If export_pixels is True, returns a tuple of two DataFrames:
                1. DataFrame with statistics for each point across all raster files.
                2. DataFrame with pixel values for each point across all raster files.
    
    Raises:
        ValueError: If input parameters are invalid.
        FileNotFoundError: If a raster file is not found.
        rasterio.errors.RasterioIOError: If there's an error reading a raster file.
    
    Notes:
        - Returns NaN for a statistic if all pixels in the window are NaN or nodata.
        - Reads nodata value from raster metadata for each file.
        
        Ehsan Farahbakhsh
        EarthByte Group, School of Geosciences, The University of Sydney, Sydney, Australia
        Date: 26/09/2024
    """
    # Convert inputs to numpy arrays if they're not already
    xs = np.array(xs)
    ys = np.array(ys)

    # Input validation
    if xs.size == 0 or ys.size == 0:
        raise ValueError("The xs and ys arrays cannot be empty.")
    if xs.shape != ys.shape:
        raise ValueError("The xs and ys arrays must have the same shape.")
    if isinstance(raster_paths, str):
        raster_paths = [raster_paths]
    if not raster_paths:
        raise ValueError("raster_paths cannot be empty.")
    if buffer_shape not in ['circle', 'square']:
        raise ValueError("buffer_shape must be either 'circle' or 'square'.")
    if buffer_size < 1:
        raise ValueError("buffer_size must be a positive integer.")
    if not stats:
        raise ValueError("The stats list cannot be empty.")
    if not set(stats).issubset({'mean', 'std', 'min', 'max', 'median'}):
        raise ValueError("Unsupported statistic in stats. Supported stats are: 'mean', 'std', 'min', 'max', 'median'.")

    results = [{} for _ in range(xs.size)]  # Initialize a list of dictionaries for each point
    pixel_values = [{} for _ in range(xs.size)]  # Initialize a list of dictionaries for pixel values
    


    for raster_path in raster_paths:
        if not os.path.exists(raster_path):
            raise FileNotFoundError(f"Raster file not found: {raster_path}")
        
        raster_name = os.path.splitext(os.path.basename(raster_path))[0]
        
        try:
            with rasterio.open(raster_path) as src:
                # Read nodata value from metadata
                nodata_value = src.nodata
                
                for i, (x, y) in enumerate(zip(xs.flat, ys.flat)):
                    # Convert geographic coordinates to pixel coordinates
                    row, col = src.index(x, y)
                    
                    # Convert row and col to integers
                    row, col = int(row), int(col)
                    
                    # Define the window around the point
                    window_row_start = max(0, row - buffer_size)
                    window_row_end = min(src.height, row + buffer_size + 1)
                    window_col_start = max(0, col - buffer_size)
                    window_col_end = min(src.width, col + buffer_size + 1)
                    
                    window = Window(window_col_start, window_row_start, 
                                    window_col_end - window_col_start, 
                                    window_row_end - window_row_start)
                    
                    # Read the data within the window
                    data = src.read(1, window=window)
                    
                    # Create the mask
                    mask_shape = (2*buffer_size+1, 2*buffer_size+1)
                    if buffer_shape == 'circle':
                        y_grid, x_grid = np.ogrid[-buffer_size:buffer_size+1, -buffer_size:buffer_size+1]
                        mask = x_grid*x_grid + y_grid*y_grid <= buffer_size*buffer_size
                    else:  # square
                        mask = np.ones(mask_shape, dtype=bool)
                    
                    # Adjust mask to match data shape
                    mask = mask[max(0, buffer_size-row):mask_shape[0]-(max(0, row+buffer_size+1-src.height)),
                                max(0, buffer_size-col):mask_shape[1]-(max(0, col+buffer_size+1-src.width))]
                    
                    # Apply the mask
                    masked_data = data[mask]
                    
                    # Remove nodata values
                    if nodata_value is not None:
                        masked_data = masked_data[masked_data != nodata_value]
                    
                    # Check if all values are NaN or if no valid data remains
                    all_invalid = np.all(np.isnan(masked_data)) or masked_data.size == 0
                    
                    # Calculate statistics
                    for stat in stats:
                        if all_invalid:
                            results[i][f'{raster_name}_{stat}'] = np.nan
                        elif stat == 'mean':
                            results[i][f'{raster_name}_{stat}'] = np.nanmean(masked_data)
                        elif stat == 'std':
                            results[i][f'{raster_name}_{stat}'] = np.nanstd(masked_data)
                        elif stat == 'min':
                            results[i][f'{raster_name}_{stat}'] = np.nanmin(masked_data)
                        elif stat == 'max':
                            results[i][f'{raster_name}_{stat}'] = np.nanmax(masked_data)
                        elif stat == 'median':
                            results[i][f'{raster_name}_{stat}'] = np.nanmedian(masked_data)
                    
                    # Store pixel values if export_pixels is True
                    if export_pixels:
                        pixel_values[i][f'{raster_name}_pixels'] = masked_data.tolist()
                        
        except rasterio.errors.RasterioIOError as e:
            raise rasterio.errors.RasterioIOError(f"Error reading raster file {raster_path}: {str(e)}")
    
    stats_df = pd.DataFrame(results)
    
    if export_pixels:
        pixels_df = pd.DataFrame(pixel_values)
        return stats_df, pixels_df
    else:
        return stats_df

# -----------

def get_grid_grad_stat_features(
    xs: Union[List[float], np.ndarray, Series],
    ys: Union[List[float], np.ndarray, Series],
    raster_paths: Union[str, List[str]],
    gradient_directions: List[str] = ['x', 'y', 'both'],
    buffer_shape: str = 'circle',
    buffer_size: int = 1,
    stats: List[str] = ['mean', 'std', 'min', 'max', 'median'],
    export_pixels: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Calculate statistics for gradients of pixels surrounding each point in each raster file.
    Optionally export gradient values for each point.
    
    Args:
        xs (Union[List[float], np.ndarray, Series]): List, array, or Series of x-coordinates.
        ys (Union[List[float], np.ndarray, Series]): List, array, or Series of y-coordinates.
        raster_paths (Union[str, List[str]]): Path to a single raster file or list of paths to georeferenced raster files.
        gradient_directions (List[str], optional): List of directions for gradient calculation.
                                                   Supported directions: 'x', 'y', 'both'.
                                                   Defaults to ['x', 'y', 'both'].
        buffer_shape (str, optional): Shape of the buffer zone ('circle' or 'square'). Defaults to 'circle'.
        buffer_size (int, optional): For 'circle', radius in pixels. For 'square', half of the side length in pixels. Defaults to 1.
        stats (List[str], optional): List of statistics to calculate. 
                                     Supported stats: 'mean', 'std', 'min', 'max', 'median'.
                                     Defaults to ['mean', 'std', 'min', 'max', 'median'].
        export_pixels (bool, optional): If True, export gradient values for each point. Defaults to False.
    
    Returns:
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]: 
            If export_pixels is False, returns a DataFrame with gradient statistics for each point across all raster files.
            If export_pixels is True, returns a tuple of two DataFrames:
                1. DataFrame with gradient statistics for each point across all raster files.
                2. DataFrame with gradient values for each point across all raster files.
    
    Raises:
        ValueError: If input parameters are invalid.
        FileNotFoundError: If a raster file is not found.
        rasterio.errors.RasterioIOError: If there's an error reading a raster file.
    
    Notes:
        - Returns NaN for a statistic if all pixels in the window are NaN or nodata.
        - Reads nodata value from raster metadata for each file.
        
        Ehsan Farahbakhsh
        EarthByte Group, School of Geosciences, The University of Sydney, Sydney, Australia
        Date: 26/09/2024
    """
    # Convert inputs to numpy arrays if they're not already
    xs = np.array(xs)
    ys = np.array(ys)

    # Input validation
    if xs.size == 0 or ys.size == 0:
        raise ValueError("The xs and ys arrays cannot be empty.")
    if xs.shape != ys.shape:
        raise ValueError("The xs and ys arrays must have the same shape.")
    if isinstance(raster_paths, str):
        raster_paths = [raster_paths]
    if not raster_paths:
        raise ValueError("raster_paths cannot be empty.")
    if buffer_shape not in ['circle', 'square']:
        raise ValueError("buffer_shape must be either 'circle' or 'square'.")
    if buffer_size < 1:
        raise ValueError("buffer_size must be a positive integer.")
    if not stats:
        raise ValueError("The stats list cannot be empty.")
    if not set(stats).issubset({'mean', 'std', 'min', 'max', 'median'}):
        raise ValueError("Unsupported statistic in stats. Supported stats are: 'mean', 'std', 'min', 'max', 'median'.")
    if not gradient_directions:
        raise ValueError("gradient_directions list cannot be empty.")
    if not set(gradient_directions).issubset({'x', 'y', 'both'}):
        raise ValueError("Unsupported gradient direction. Supported directions are: 'x', 'y', 'both'.")

    results = [{} for _ in range(xs.size)]  # Initialize a list of dictionaries for each point
    gradient_values = [{} for _ in range(xs.size)]  # Initialize a list of dictionaries for gradient values
    
    for raster_path in raster_paths:
        if not os.path.exists(raster_path):
            raise FileNotFoundError(f"Raster file not found: {raster_path}")
        
        raster_name = os.path.splitext(os.path.basename(raster_path))[0]
        
        try:
            with rasterio.open(raster_path) as src:
                # Read nodata value from metadata
                nodata_value = src.nodata
                
                for i, (x, y) in enumerate(zip(xs.flat, ys.flat)):
                    # Convert geographic coordinates to pixel coordinates
                    row, col = src.index(x, y)
                    
                    # Convert row and col to integers
                    row, col = int(row), int(col)
                    
                    # Define the window around the point
                    window_row_start = max(0, row - buffer_size)
                    window_row_end = min(src.height, row + buffer_size + 1)
                    window_col_start = max(0, col - buffer_size)
                    window_col_end = min(src.width, col + buffer_size + 1)
                    
                    window = Window(window_col_start, window_row_start, 
                                    window_col_end - window_col_start, 
                                    window_row_end - window_row_start)
                    
                    # Read the data within the window
                    data = src.read(1, window=window)
                    
                    # Create the mask
                    mask_shape = (2*buffer_size+1, 2*buffer_size+1)
                    if buffer_shape == 'circle':
                        y_grid, x_grid = np.ogrid[-buffer_size:buffer_size+1, -buffer_size:buffer_size+1]
                        mask = x_grid*x_grid + y_grid*y_grid <= buffer_size*buffer_size
                    else:  # square
                        mask = np.ones(mask_shape, dtype=bool)
                    
                    # Adjust mask to match data shape
                    mask = mask[max(0, buffer_size-row):mask_shape[0]-(max(0, row+buffer_size+1-src.height)),
                                max(0, buffer_size-col):mask_shape[1]-(max(0, col+buffer_size+1-src.width))]
                    
                    # Remove nodata values
                    if nodata_value is not None:
                        mask = mask & (data != nodata_value)
                    
                    # Calculate gradients
                    gradients = {}
                    if 'x' in gradient_directions or 'both' in gradient_directions:
                        gradients['x'] = np.gradient(data, axis=1)
                    if 'y' in gradient_directions or 'both' in gradient_directions:
                        gradients['y'] = np.gradient(data, axis=0)
                    if 'both' in gradient_directions:
                        gradients['both'] = np.sqrt(gradients['x']**2 + gradients['y']**2)
                    
                    for direction, gradient in gradients.items():
                        # Apply the mask
                        masked_gradient = gradient[mask]
                        
                        # Check if all values are NaN or if no valid data remains
                        all_invalid = np.all(np.isnan(masked_gradient)) or masked_gradient.size == 0
                        
                        # Calculate statistics
                        for stat in stats:
                            if all_invalid:
                                results[i][f'{raster_name}_{direction}_{stat}'] = np.nan
                            elif stat == 'mean':
                                results[i][f'{raster_name}_{direction}_{stat}'] = np.nanmean(masked_gradient)
                            elif stat == 'std':
                                results[i][f'{raster_name}_{direction}_{stat}'] = np.nanstd(masked_gradient)
                            elif stat == 'min':
                                results[i][f'{raster_name}_{direction}_{stat}'] = np.nanmin(masked_gradient)
                            elif stat == 'max':
                                results[i][f'{raster_name}_{direction}_{stat}'] = np.nanmax(masked_gradient)
                            elif stat == 'median':
                                results[i][f'{raster_name}_{direction}_{stat}'] = np.nanmedian(masked_gradient)
                        
                        # Store gradient values if export_pixels is True
                        if export_pixels:
                            gradient_values[i][f'{raster_name}_{direction}_gradient'] = masked_gradient.tolist()
                        
        except rasterio.errors.RasterioIOError as e:
            raise rasterio.errors.RasterioIOError(f"Error reading raster file {raster_path}: {str(e)}")
    
    stats_df = pd.DataFrame(results)
    
    if export_pixels:
        gradient_df = pd.DataFrame(gradient_values)
        return stats_df, gradient_df
    else:
        return stats_df

# -----------

def get_grid_tex_features(
    xs: Union[List[float], np.ndarray, Series],
    ys: Union[List[float], np.ndarray, Series],
    raster_paths: Union[str, List[str]],
    buffer_size: int = 1,
    texture_features: List[str] = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'],
    export_pixels: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Calculate GLCM texture features for pixels surrounding each point in each raster file.
    Optionally export pixel values for each point.
    
    Args:
        xs (Union[List[float], np.ndarray, Series]): List, array, or Series of x-coordinates.
        ys (Union[List[float], np.ndarray, Series]): List, array, or Series of y-coordinates.
        raster_paths (Union[str, List[str]]): Path to a single raster file or list of paths to georeferenced raster files.
        buffer_size (int, optional): Half of the side length of the square buffer in pixels. Defaults to 1.
        texture_features (List[str], optional): List of GLCM texture features to calculate. 
                                                Supported features: 'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'.
                                                Defaults to ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'].
        export_pixels (bool, optional): If True, export pixel values for each point. Defaults to False.
    
    Returns:
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]: 
            If export_pixels is False, returns a DataFrame with texture features for each point across all raster files.
            If export_pixels is True, returns a tuple of two DataFrames:
                1. DataFrame with texture features for each point across all raster files.
                2. DataFrame with pixel values for each point across all raster files.
    
    Raises:
        ValueError: If input parameters are invalid.
        FileNotFoundError: If a raster file is not found.
        rasterio.errors.RasterioIOError: If there's an error reading a raster file.
    
    Notes:
        - Returns NaN for a feature if all pixels in the window are nodata or invalid.
        - Reads nodata value from raster metadata for each file.
        
        Ehsan Farahbakhsh
        EarthByte Group, School of Geosciences, The University of Sydney, Sydney, Australia
        Date: 11/09/2024
    """
    # Convert inputs to numpy arrays if they're not already
    xs = np.array(xs)
    ys = np.array(ys)

    # Input validation
    if xs.size == 0 or ys.size == 0:
        raise ValueError("The xs and ys arrays cannot be empty.")
    if xs.shape != ys.shape:
        raise ValueError("The xs and ys arrays must have the same shape.")
    if isinstance(raster_paths, str):
        raster_paths = [raster_paths]
    if not raster_paths:
        raise ValueError("raster_paths cannot be empty.")
    if buffer_size < 1:
        raise ValueError("buffer_size must be a positive integer.")
    if not texture_features:
        raise ValueError("The texture_features list cannot be empty.")
    if not set(texture_features).issubset({'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'}):
        raise ValueError("Unsupported texture feature. Supported features are: 'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'.")

    results = [{} for _ in range(xs.size)]  # Initialize a list of dictionaries for each point
    pixel_values = [{} for _ in range(xs.size)]  # Initialize a list of dictionaries for pixel values
    
    for raster_path in raster_paths:
        if not os.path.exists(raster_path):
            raise FileNotFoundError(f"Raster file not found: {raster_path}")
        
        raster_name = os.path.splitext(os.path.basename(raster_path))[0]
        
        try:
            with rasterio.open(raster_path) as src:
                # Read nodata value from metadata
                nodata_value = src.nodata
                
                for i, (x, y) in enumerate(zip(xs.flat, ys.flat)):
                    # Convert geographic coordinates to pixel coordinates
                    row, col = src.index(x, y)
                    
                    # Define the window around the point, handling edge cases
                    window_row_start = max(0, row - buffer_size)
                    window_row_end = min(src.height, row + buffer_size + 1)
                    window_col_start = max(0, col - buffer_size)
                    window_col_end = min(src.width, col + buffer_size + 1)
                    
                    window = Window(window_col_start, window_row_start, 
                                    window_col_end - window_col_start, 
                                    window_row_end - window_row_start)
                    
                    # Read the data within the window
                    data = src.read(1, window=window)
                    
                    # Create a mask for nodata values
                    if nodata_value is not None:
                        mask = (data != nodata_value)
                    else:
                        mask = np.ones_like(data, dtype=bool)
                    
                    # Check if all values are invalid or if no valid data remains
                    all_invalid = np.all(~mask) or data.size == 0
                    
                    if not all_invalid:
                        # Use only valid data for normalization
                        valid_data = data[mask]
                        
                        # Normalize the data to 0-255 range for GLCM
                        data_min, data_max = np.min(valid_data), np.max(valid_data)
                        if data_max > data_min:
                            data_normalized = np.where(mask, ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8), 0)
                        else:
                            # If all valid values are the same, set them to 255
                            data_normalized = np.where(mask, 255, 0)
                        
                        # Calculate GLCM
                        glcm = graycomatrix(data_normalized, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
                        
                        # Calculate texture features
                        for feature in texture_features:
                            feature_value = np.mean(graycoprops(glcm, feature)[0])
                            results[i][f'{raster_name}_{feature}'] = feature_value
                    else:
                        for feature in texture_features:
                            results[i][f'{raster_name}_{feature}'] = np.nan
                    
                    # Store pixel values if export_pixels is True
                    if export_pixels:
                        pixel_values[i][f'{raster_name}_pixels'] = data.tolist()
                        
        except rasterio.errors.RasterioIOError as e:
            raise rasterio.errors.RasterioIOError(f"Error reading raster file {raster_path}: {str(e)}")
    
    texture_df = pd.DataFrame(results)
    
    if export_pixels:
        pixels_df = pd.DataFrame(pixel_values)
        return texture_df, pixels_df
    else:
        return texture_df
