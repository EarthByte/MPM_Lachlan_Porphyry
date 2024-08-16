import geopandas as gpd
import numpy as np
import os
import pandas as pd
from pyproj import CRS, Geod
import rioxarray as rxr
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import nearest_points
from shapely.strtree import STRtree
from skimage import exposure, util
from skimage.feature import graycomatrix, graycoprops
from typing import List, Literal, Union

# -------------------------------------

def get_ellipsoid_name(crs):
    """Helper function to get the correct ellipsoid name for Geod"""
    ellipsoid_name = crs.ellipsoid.name.lower()
    if 'grs' in ellipsoid_name and '1980' in ellipsoid_name:
        return 'GRS80'
    elif 'wgs' in ellipsoid_name and '84' in ellipsoid_name:
        return 'WGS84'
    else:
        return 'WGS84'  # Default to WGS84 if unknown

def get_nearest_point(point, geometry):
    """Helper function to get the nearest point on a geometry"""
    if isinstance(geometry, (LineString, MultiLineString)):
        return nearest_points(point, geometry)[1]
    return geometry

def get_suitable_projected_crs(gdf):
    """Helper function to get a suitable projected CRS based on the data's extent"""
    bounds = gdf.total_bounds
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2
    utm_zone = int((center_lon + 180) / 6) + 1
    epsg = 32600 + utm_zone + (0 if center_lat >= 0 else 100)
    return CRS(f"EPSG:{epsg}")

def get_dist_line(xs: Union[List[float], np.ndarray], 
                  ys: Union[List[float], np.ndarray], 
                  line_files: List[str], 
                  distance_type: Literal['euclidean', 'geodesic'] = 'euclidean',
                  input_crs: str = 'EPSG:4283') -> pd.DataFrame:
    """
    Calculate the distance from points to linear features.
    Args:
    xs (Union[List[float], np.ndarray]): x-coordinates of the points
    ys (Union[List[float], np.ndarray]): y-coordinates of the points
    line_files (List[str]): a list of file paths to linear features
    distance_type (str): type of distance calculation ('euclidean' or 'geodesic')
    input_crs (str): Coordinate Reference System of input data
    Returns:
    pd.DataFrame: A DataFrame where each column represents distances to a line in meters
    """
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    
    if distance_type not in ['euclidean', 'geodesic']:
        raise ValueError("distance_type must be either 'euclidean' or 'geodesic'")

    lines = []
    column_names = []
    
    input_crs = CRS(input_crs)
    
    for file in line_files:
        gdf = gpd.read_file(file)
        if gdf.crs is None:
            gdf.set_crs(input_crs, inplace=True)
        lines.append(gdf)
        column_names.append(os.path.splitext(os.path.basename(file))[0])
    
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
        ellps = get_ellipsoid_name(points_gdf.crs)
        geod = Geod(ellps=ellps)
    
    for i, line in enumerate(lines):
        line_tree = STRtree(line.geometry)
        nearest_geoms = [line_tree.nearest(p) for p in points_gdf.geometry]
        
        if distance_type == 'euclidean':
            dist_to_lines[:, i] = [p.distance(nearest) for p, nearest in zip(points_gdf.geometry, nearest_geoms)]
        else:  # geodesic
            for j, (p, nearest) in enumerate(zip(points_gdf.geometry, nearest_geoms)):
                nearest_point = get_nearest_point(p, nearest)
                _, _, distance = geod.inv(p.x, p.y, nearest_point.x, nearest_point.y)
                dist_to_lines[j, i] = distance
    
    return pd.DataFrame(dist_to_lines, columns=column_names)
# ----------------------------------------------------------

def get_cat_data(
    xs: Union[List[float], np.ndarray],
    ys: Union[List[float], np.ndarray],
    polygon_files: Union[str, List[str]],
    field: str
) -> pd.DataFrame:
    """
    Extract categorical data from geographic polygon files based on given point coordinates.

    This function performs a spatial join between input points and polygons from specified files,
    extracting a given field value for each point based on the polygon it falls within.

    Parameters:
    -----------
    xs : Union[List[float], np.ndarray]
        List or array of x-coordinates.
    ys : Union[List[float], np.ndarray]
        List or array of y-coordinates.
    polygon_files : Union[str, List[str]]
        Single file path or list of file paths to polygon shapefiles.
    field : str
        Name of the field in the polygon files to extract.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the extracted categorical data for each point and each polygon file.
        Columns are named after the input files, rows correspond to input points.

    Raises:
    -------
    ValueError
        If input coordinates are not of equal length or if polygon_files is empty.
    FileNotFoundError
        If any of the specified polygon files does not exist.

    Notes:
    ------
    - Points that do not fall within any polygon will have 'Null' as their value.
    - Any errors during file processing will be printed, and the file will be skipped.
    """
    # Input validation
    if len(xs) != len(ys):
        raise ValueError("Input coordinates must be of equal length")
    
    # Convert polygon_files to a list if it's a single string
    if isinstance(polygon_files, str):
        polygon_files = [polygon_files]
    
    if not polygon_files:
        raise ValueError("At least one polygon file must be specified")
    
    for file in polygon_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Polygon file not found: {file}")

    # Create GeoDataFrame from input points
    points_gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(xs, ys)])
    cat_code = pd.DataFrame()

    for file in polygon_files:
        column_name = os.path.splitext(os.path.basename(file))[0]
        try:
            polygon_gdf = gpd.read_file(file)
            
            # Ensure the field exists in the GeoDataFrame
            if field not in polygon_gdf.columns:
                print(f"Warning: Field '{field}' not found in {column_name}. Skipping.")
                continue
            
            # Spatial join
            joined = gpd.sjoin(points_gdf, polygon_gdf, how="left", predicate="within")
            
            # Extract the field values, replacing NaN with 'Null'
            cat_code[column_name] = joined[field].fillna('Null')
            
        except Exception as e:
            print(f"Error processing {column_name}: {str(e)}")
    
    return cat_code
# -----------------

def get_grid_stat_features(
    xs: Union[List[float], np.ndarray],
    ys: Union[List[float], np.ndarray],
    grid_paths: Union[str, List[str]],
    buffer_size: float,
    buffer_shape: str = 'circular',
    stats: List[str] = ['mean', 'std'],
    gradient: Union[None, str, List[str]] = None
) -> pd.DataFrame:
    """
    Calculate specified statistics within a buffer around given points for one or multiple raster files.

    Args:
        xs (Union[List[float], np.ndarray]): X coordinates of the points.
        ys (Union[List[float], np.ndarray]): Y coordinates of the points.
        grid_paths (Union[str, List[str]]): Path(s) to the raster file(s).
        buffer_size (float): Size of the buffer in meters (radius for circular, half side length for square).
        buffer_shape (str, optional): Shape of the buffer, either 'circular' or 'square'. Defaults to 'circular'.
        stats (List[str], optional): List of statistics to calculate. Defaults to ['mean', 'std'].
        gradient (Union[None, str, List[str]], optional): Direction(s) to calculate gradient, can be None, 'x', 'y', 'both', or ['x', 'y']. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with specified statistic columns for each input raster and gradient type.

    Raises:
        ValueError: If buffer_shape is not 'circular' or 'square'.
        ValueError: If the number of x and y coordinates don't match.
        FileNotFoundError: If any of the specified raster files are not found.
    """
    # Input validation
    if buffer_shape not in ['circular', 'square']:
        raise ValueError("buffer_shape must be either 'circular' or 'square'")
    
    if len(xs) != len(ys):
        raise ValueError("The number of x and y coordinates must be the same")

    valid_gradients = [None, 'x', 'y', 'both', ['x', 'y']]
    if gradient not in valid_gradients:
        raise ValueError("gradient must be None, 'x', 'y', 'both', or ['x', 'y']")

    # Convert inputs to numpy arrays
    xs = np.array(xs)
    ys = np.array(ys)

    # Ensure grid_paths is a list
    if isinstance(grid_paths, str):
        grid_paths = [grid_paths]

    # Check if all files exist
    for path in grid_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raster file not found: {path}")

    # Define statistic functions
    stat_functions = {
        'mean': np.nanmean,
        'std': np.nanstd,
        'min': np.nanmin,
        'max': np.nanmax,
        'median': np.nanmedian
    }

    results = {}

    for grid_path in grid_paths:
        # Open the raster file
        raster = rxr.open_rasterio(grid_path, masked=True).squeeze()
        bounds = raster.rio.bounds()
        raster_np = raster.values

        # Get the CRS of the raster
        raster_crs = raster.rio.crs

        # Prepare gradient rasters
        rasters_to_process = [('original', raster_np)]
        if gradient:
            if gradient in ['x', 'both'] or (isinstance(gradient, list) and 'x' in gradient):
                grad_x = np.gradient(raster_np, axis=1)
                rasters_to_process.append(('grad_x', grad_x))
            if gradient in ['y', 'both'] or (isinstance(gradient, list) and 'y' in gradient):
                grad_y = np.gradient(raster_np, axis=0)
                rasters_to_process.append(('grad_y', grad_y))
            if gradient == 'both':
                grad_both = np.sqrt(grad_x**2 + grad_y**2)
                rasters_to_process.append(('grad_both', grad_both))

        # Convert coordinates to pixel space
        x_pixels = ((xs - bounds[0]) / (bounds[2] - bounds[0]) * (raster_np.shape[1] - 1)).astype(int)
        y_pixels = ((ys - bounds[1]) / (bounds[3] - bounds[1]) * (raster_np.shape[0] - 1)).astype(int)

        # Calculate pixel size
        if raster_crs.is_projected:
            pixel_size_x = (bounds[2] - bounds[0]) / raster_np.shape[1] * raster_crs.linear_units_factor
            pixel_size_y = (bounds[3] - bounds[1]) / raster_np.shape[0] * raster_crs.linear_units_factor
        else:
            # Approximate pixel size for geographic CRS
            pixel_size_x = (bounds[2] - bounds[0]) / raster_np.shape[1] * 111319.9  # approx meters per degree at equator
            pixel_size_y = (bounds[3] - bounds[1]) / raster_np.shape[0] * 111319.9

        # Calculate buffer size in pixels
        pixels_per_meter = min(1/pixel_size_x, 1/pixel_size_y)
        buffer_pixels = max(int(buffer_size * pixels_per_meter), 1)  # Ensure at least 1 pixel

        for raster_type, raster_data in rasters_to_process:
            stat_results = {stat: [] for stat in stats}

            if buffer_shape == 'square':
                for x, y in zip(x_pixels, y_pixels):
                    x_min, x_max = max(0, x - buffer_pixels), min(raster_data.shape[1], x + buffer_pixels + 1)
                    y_min, y_max = max(0, y - buffer_pixels), min(raster_data.shape[0], y + buffer_pixels + 1)
                    
                    values = raster_data[y_min:y_max, x_min:x_max]
                    valid_values = values[~np.isnan(values) & ~np.isinf(values)]
                    for stat in stats:
                        if valid_values.size > 0:
                            stat_results[stat].append(stat_functions[stat](valid_values))
                        else:
                            stat_results[stat].append(np.nan)
            else:  # circular buffer
                y_indices, x_indices = np.indices(raster_data.shape)
                for x, y in zip(x_pixels, y_pixels):
                    mask = (x_indices - x)**2 + (y_indices - y)**2 <= buffer_pixels**2
                    values = raster_data[mask]
                    valid_values = values[~np.isnan(values) & ~np.isinf(values)]
                    for stat in stats:
                        if valid_values.size > 0:
                            stat_results[stat].append(stat_functions[stat](valid_values))
                        else:
                            stat_results[stat].append(np.nan)

            # Create a prefix for column names
            prefix = os.path.splitext(os.path.basename(grid_path))[0]
            for stat in stats:
                results[f'{prefix}_{raster_type}_{stat}'] = stat_results[stat]

    # Create a DataFrame with the results
    df = pd.DataFrame(results)
    
    return df
# -----------

def get_grid_tex_features(
    xs: Union[List[float], np.ndarray],
    ys: Union[List[float], np.ndarray],
    grid_paths: Union[str, List[str]],
    buffer_size: float,
    features: List[str] = ['dissimilarity', 'correlation']
) -> pd.DataFrame:
    """
    Calculate texture features from raster grid(s) for given points using a square buffer.

    Args:
        xs (Union[List[float], np.ndarray]): X-coordinates of target points.
        ys (Union[List[float], np.ndarray]): Y-coordinates of target points.
        grid_paths (Union[str, List[str]]): Path(s) to raster file(s).
        buffer_size (float): Size of the buffer in meters (half side length of the square).
        features (List[str]): List of texture features to calculate. Options include 'contrast', 
                              'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'.
                              Defaults to ['dissimilarity', 'correlation'].

    Returns:
        pd.DataFrame: DataFrame containing selected texture feature values for each point and raster.

    Raises:
        ValueError: If the number of x and y coordinates don't match.
        FileNotFoundError: If any of the specified raster files are not found.
        ValueError: If an invalid feature is specified.
    """
    if len(xs) != len(ys):
        raise ValueError("The number of x and y coordinates must be the same")

    valid_features = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    invalid_features = [f for f in features if f not in valid_features]
    if invalid_features:
        raise ValueError(f"Invalid feature(s) specified: {', '.join(invalid_features)}. "
                         f"Valid options are: {', '.join(valid_features)}")

    # Convert inputs to numpy arrays
    xs = np.array(xs)
    ys = np.array(ys)

    # Ensure grid_paths is a list
    if isinstance(grid_paths, str):
        grid_paths = [grid_paths]

    # Check if all files exist
    for path in grid_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raster file not found: {path}")

    results = {}

    for grid_path in grid_paths:
        # Open the raster file
        raster = rxr.open_rasterio(grid_path, masked=True).squeeze()
        bounds = raster.rio.bounds()
        raster_np = raster.values

        # Get the CRS of the raster
        raster_crs = raster.rio.crs

        # Convert coordinates to pixel space
        x_pixels = ((xs - bounds[0]) / (bounds[2] - bounds[0]) * (raster_np.shape[1] - 1)).astype(int)
        y_pixels = ((ys - bounds[1]) / (bounds[3] - bounds[1]) * (raster_np.shape[0] - 1)).astype(int)

        # Calculate pixel size
        if raster_crs.is_projected:
            pixel_size_x = (bounds[2] - bounds[0]) / raster_np.shape[1] * raster_crs.linear_units_factor
            pixel_size_y = (bounds[3] - bounds[1]) / raster_np.shape[0] * raster_crs.linear_units_factor
        else:
            # Approximate pixel size for geographic CRS
            pixel_size_x = (bounds[2] - bounds[0]) / raster_np.shape[1] * 111319.9  # approx meters per degree at equator
            pixel_size_y = (bounds[3] - bounds[1]) / raster_np.shape[0] * 111319.9

        # Calculate buffer size in pixels
        pixels_per_meter = min(1/pixel_size_x, 1/pixel_size_y)
        buffer_pixels = max(int(buffer_size * pixels_per_meter), 1)  # Ensure at least 1 pixel

        feature_results = {feature: [] for feature in features}

        for x, y in zip(x_pixels, y_pixels):
            x_min, x_max = max(0, x - buffer_pixels), min(raster_np.shape[1], x + buffer_pixels + 1)
            y_min, y_max = max(0, y - buffer_pixels), min(raster_np.shape[0], y + buffer_pixels + 1)
            
            values = raster_np[y_min:y_max, x_min:x_max]
            if not np.isnan(values).all():
                raster_scaled = exposure.rescale_intensity(values, in_range=(np.nanmin(values), np.nanmax(values)), out_range=(0, 1))
                raster_ubyte = util.img_as_ubyte(raster_scaled)
                glcm = graycomatrix(raster_ubyte, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
                for feature in features:
                    feature_results[feature].append(graycoprops(glcm, feature)[0, 0])
            else:
                for feature in features:
                    feature_results[feature].append(np.nan)

        # Create a prefix for column names
        prefix = os.path.splitext(os.path.basename(grid_path))[0]
        for feature in features:
            results[f'{prefix}_{feature}'] = feature_results[feature]

    # Create a DataFrame with the results
    df = pd.DataFrame(results)
    
    return df
