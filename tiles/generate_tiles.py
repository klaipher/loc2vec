import geopandas as gpd
import psycopg2
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box, Polygon
from pyproj import Transformer
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import warnings
from typing import Dict, Tuple
import utils
warnings.filterwarnings("ignore")


def connect_to_postgis(
        dbname: str,
        user: str,
        password: str,
        host: str = "localhost",
        port: int = 5432,
) -> psycopg2.extensions.connection:
    """
    Connect to a PostGIS database using psycopg2.

    Parameters
    ----------
    dbname : str
        Name of the database.
    user : str
        Username to connect with.
    password : str
        Password for the database user.
    host : str, optional
        Database host address, by default "localhost".
    port : int, optional
        Port number for the database, by default 5432.

    Returns
    -------
    psycopg2.extensions.connection
        A psycopg2 connection object to the PostGIS database.
    """
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
    return conn


def query_layer(conn: psycopg2.extensions.connection, query: str) -> gpd.GeoDataFrame:
    """
    Query a PostGIS layer and return the results as a GeoDataFrame.

    Parameters
    ----------
    conn : psycopg2.extensions.connection
        Psycopg2 connection object pointing to a valid PostGIS instance.
    query : str
        SQL query to be executed.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the queried geometries and attributes.
    """
    gdf = gpd.read_postgis(query, conn, geom_col="way")
    return gdf


def rasterize_layer(
        gdf: gpd.GeoDataFrame,
        bounds: Polygon,
        resolution: Tuple[int, int] = (224, 224),
        out_shape: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Rasterize geometries in a GeoDataFrame within specified bounds.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing the geometries to rasterize.
    bounds : Polygon
        The polygon representing the bounding box in EPSG:3857 coordinates.
    resolution : tuple of int, optional
        The raster resolution (width, height), by default (224, 224).
    out_shape : tuple of int, optional
        The shape of the output raster array (height, width), by default (224, 224).

    Returns
    -------
    np.ndarray
        A 2D NumPy array representing the rasterized geometry with shape defined by ``out_shape``.
    """
    minx, miny, maxx, maxy = bounds.bounds
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, *resolution)

    raster = rasterize(
        [(geometry, 1) for geometry in gdf.geometry],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    return raster


def create_bounds(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
) -> Polygon:
    """
    Create a bounding box polygon in EPSG:3857 projection from latitude-longitude values.

    Parameters
    ----------
    lat1 : float
        Latitude of the first corner (e.g., upper-left corner).
    lon1 : float
        Longitude of the first corner (e.g., upper-left corner).
    lat2 : float
        Latitude of the second corner (e.g., lower-right corner).
    lon2 : float
        Longitude of the second corner (e.g., lower-right corner).

    Returns
    -------
    Polygon
        A Shapely ``Polygon`` representing the bounding box in EPSG:3857.
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    transformed_lon1, transformed_lat1 = transformer.transform(lon1, lat1)
    transformed_lon2, transformed_lat2 = transformer.transform(lon2, lat2)
    return box(transformed_lon1, transformed_lat2, transformed_lon2, transformed_lat1)


def create_multi_channel_tensor(
        bounds: Polygon,
        layers: Dict[str, gpd.GeoDataFrame],
) -> np.array:
    """
    Rasterize multiple layers to create a multi-channel tensor, then save it to disk.

    Parameters
    ----------
    bounds : Polygon
        Bounding box representing the area of interest in EPSG:3857.
    layers : dict of str to gpd.GeoDataFrame
        Dictionary where keys are layer names and values are the corresponding GeoDataFrames.
    save_path : str
        Filepath to save the resulting multi-channel NumPy array.
    """
    channels = []
    pbar = tqdm(layers.items(), desc="Rasterizing layers", total=len(layers))

    for layer_name, gdf in pbar:
        pbar.set_postfix({"Layer": layer_name})
        raster = rasterize_layer(gdf, bounds)
        channels.append(raster)

    return np.stack(channels, axis=0)


def visualize_tensor_channels(tensor: np.ndarray) -> None:
    """
    Visualize each channel of a 3D tensor (n_channels, height, width).

    Parameters
    ----------
    tensor : np.ndarray
        A 3D NumPy array of shape (n_channels, height, width).
    """
    n_channels = tensor.shape[0]
    fig, axes = plt.subplots(1, n_channels, figsize=(15, 5))

    for i in range(n_channels):
        axes[i].imshow(tensor[i], cmap='gray')
        axes[i].set_title(f'Channel {i + 1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def rasterize_postgis_data(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        output_folder: str,
        dbname: str,
        user: str,
        password: str,
        save_path: str = "rasterized_tensor.npy",
) -> np.array:
    """
    Connect to PostGIS, define a bounding box, query OSM layers, rasterize them,
    create a multi-channel tensor, and save it.

    Parameters
    ----------
    lat1 : float
        Latitude of the first bounding box corner (e.g., upper-left).
    lon1 : float
        Longitude of the first bounding box corner (e.g., upper-left).
    lat2 : float
        Latitude of the second bounding box corner (e.g., lower-right).
    lon2 : float
        Longitude of the second bounding box corner (e.g., lower-right).
    output_folder : str
        Path of the folder where the output .npy file will be saved.
    dbname : str
        Name of the PostGIS database.
    user : str
        Username for the PostGIS database.
    password : str
        Password for the PostGIS database.
    """
    conn = connect_to_postgis(dbname, user, password)
    bounds = create_bounds(lat1, lon1, lat2, lon2)
    lon1_3857, lat1_3857, lon2_3857, lat2_3857 = bounds.bounds

    queries = {
        "roads": f"""
            SELECT way
            FROM planet_osm_line 
            WHERE highway IN ('motorway', 'primary', 'trunk', 'secondary', 'minor street')
            AND ST_Intersects(way, ST_MakeEnvelope({lon1_3857}, {lat1_3857}, {lon2_3857}, {lat2_3857}, 3857));
        """,
        "other_roads": f"""
            SELECT way
            FROM planet_osm_line 
            WHERE highway NOT IN ('motorway', 'primary', 'trunk', 'secondary', 'minor street')
            AND ST_Intersects(way, ST_MakeEnvelope({lon1_3857}, {lat1_3857}, {lon2_3857}, {lat2_3857}, 3857));
        """,
        "water": f"""
            SELECT way
            FROM planet_osm_polygon as t
            WHERE (t.natural = 'water' OR waterway = 'river')
            AND ST_Intersects(way, ST_MakeEnvelope({lon1_3857}, {lat1_3857}, {lon2_3857}, {lat2_3857}, 3857));
        """,
        "rails": f"""
            SELECT way
            FROM planet_osm_line
            WHERE railway IN ('rail', 'others')
            AND ST_Intersects(way, ST_MakeEnvelope({lon1_3857}, {lat1_3857}, {lon2_3857}, {lat2_3857}, 3857));
        """,
        "parks_and_forests": f"""
            SELECT way
            FROM planet_osm_polygon
            WHERE (leisure IN ('park') OR landuse IN ('forest', 'grass', 'meadow', 'orchard', 'national park'))
            AND ST_Intersects(way, ST_MakeEnvelope({lon1_3857}, {lat1_3857}, {lon2_3857}, {lat2_3857}, 3857));
        """,
        "agriculture": f"""
            SELECT way
            FROM planet_osm_polygon
            WHERE landuse IN ('farmland', 'farmyard')
            AND ST_Intersects(way, ST_MakeEnvelope({lon1_3857}, {lat1_3857}, {lon2_3857}, {lat2_3857}, 3857));
        """,
        "residential": f"""
            SELECT way
            FROM planet_osm_polygon
            WHERE landuse = 'residential'
            AND ST_Intersects(way, ST_MakeEnvelope({lon1_3857}, {lat1_3857}, {lon2_3857}, {lat2_3857}, 3857));
        """,
        "apartments": f"""
            SELECT way
            FROM planet_osm_polygon
            WHERE building IN ('apartments', 'yes')
            AND ST_Intersects(way, ST_MakeEnvelope({lon1_3857}, {lat1_3857}, {lon2_3857}, {lat2_3857}, 3857));
        """,
        "commercial": f"""
            SELECT way
            FROM planet_osm_polygon
            WHERE landuse = 'commercial'
            AND ST_Intersects(way, ST_MakeEnvelope({lon1_3857}, {lat1_3857}, {lon2_3857}, {lat2_3857}, 3857));
        """,
        "industrial_and_construction": f"""
            SELECT way
            FROM planet_osm_polygon
            WHERE landuse IN ('industrial', 'construction', 'military')
            AND ST_Intersects(way, ST_MakeEnvelope({lon1_3857}, {lat1_3857}, {lon2_3857}, {lat2_3857}, 3857));
        """,
        "educational": f"""
            SELECT way
            FROM planet_osm_polygon
            WHERE amenity IN ('school', 'university', 'college', 'kindergarten')
            AND ST_Intersects(way, ST_MakeEnvelope({lon1_3857}, {lat1_3857}, {lon2_3857}, {lat2_3857}, 3857));
        """,
        "retail": f"""
            SELECT way
            FROM planet_osm_polygon
            WHERE landuse = 'retail'
            AND ST_Intersects(way, ST_MakeEnvelope({lon1_3857}, {lat1_3857}, {lon2_3857}, {lat2_3857}, 3857));
        """,
        "other_buildings": f"""
            SELECT way
            FROM planet_osm_polygon
            WHERE landuse IN ('allotments', 'cemetery')
            AND ST_Intersects(way, ST_MakeEnvelope({lon1_3857}, {lat1_3857}, {lon2_3857}, {lat2_3857}, 3857));
        """,
        "garages_and_parking": f"""
            SELECT way
            FROM planet_osm_polygon
            WHERE (building = 'garage' OR amenity = 'parking')
            AND ST_Intersects(way, ST_MakeEnvelope({lon1_3857}, {lat1_3857}, {lon2_3857}, {lat2_3857}, 3857));
        """
    }

    layers = {}
    pbar = tqdm(queries.items(), desc="Querying layers", total=len(queries))
    for layer_name, sql_query in pbar:
        pbar.set_postfix({"Layer": layer_name})
        layers[layer_name] = query_layer(conn, sql_query)

    save_path = os.path.join(output_folder, "rasterized_tensor.npy")
    tile_tensor = create_multi_channel_tensor(bounds, layers)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, np.stack(tile_tensor, axis=0))
    print(f"Tensor saved to {save_path}")

    conn.close()

    return tile_tensor



if __name__ == "__main__":
    lat1_val, lon1_val = 50.383018073266285, 30.444404923939615  # Upper-left corner
    lat2_val, lon2_val = 50.3748645582537, 30.4618999860186      # Lower-right corner

    output_dir = "./data/output"
    db_name = "gis"
    db_user = "renderer"
    db_password = "renderer"

    tensor = rasterize_postgis_data(
        lat1_val, lon1_val, lat2_val, lon2_val,
        output_dir, db_name, db_user, db_password,
    )

    visualize_tensor_channels(tensor)
