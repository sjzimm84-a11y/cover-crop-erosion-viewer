import io
import tempfile
import zipfile
from pathlib import Path
from typing import Union

import geopandas as gpd


def save_uploaded_file(uploaded_file: "streamlit.runtime.uploaded_file_manager.UploadedFile", temp_dir: Path) -> str:
    target_path = temp_dir / uploaded_file.name
    with open(target_path, "wb") as handle:
        handle.write(uploaded_file.getvalue())
    return str(target_path)


def extract_shapefile_zip(zip_path: str) -> str:
    extract_dir = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)

    for suffix in [".shp", ".geojson", ".json"]:
        files = list(extract_dir.rglob(f"*{suffix}"))
        if files:
            return str(files[0])

    raise ValueError("No shapefile or GeoJSON was found inside the uploaded ZIP.")


def load_boundary_file(path: str) -> gpd.GeoDataFrame:
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".zip":
        path_to_vector = extract_shapefile_zip(path)
    else:
        path_to_vector = path

    read_kwargs = {}
    if Path(path_to_vector).suffix.lower() == ".kml":
        read_kwargs["driver"] = "KML"
    boundary = gpd.read_file(path_to_vector, **read_kwargs)

    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:4326")
    if len(boundary) > 1:
        boundary = boundary.dissolve().reset_index(drop=True)
    return boundary


def read_vector_file(gdf: gpd.GeoDataFrame) -> dict:
    return gdf.to_crs("EPSG:4326").__geo_interface__
