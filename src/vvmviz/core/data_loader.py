"""
VVM Data Loading Module

Handles loading VVM simulation data with caching and thread safety.
Provides functions for accessing datasets, terrain data, and simulation metadata.
"""

import os
import re
import logging
from glob import glob
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Any
import xarray as xr

import vvm_reader as vvm

from vvmviz.config import (
    FILE_IO_LOCK,
    DEFAULT_VVM_DIR,
    DATASET_CACHE_SIZE,
    TERRAIN_VAR_NAME,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Simulation Discovery
# =============================================================================

def list_simulations(base_path: Path | str = DEFAULT_VVM_DIR) -> List[Path]:
    """
    List all available VVM simulations in a directory.

    Parameters
    ----------
    base_path : Path or str
        Base directory containing VVM simulation directories

    Returns
    -------
    list of Path
        List of simulation paths found

    Examples
    --------
    >>> sims = list_simulations('/data2/VVM/taiwanvvm_summer/')
    >>> print(f"Found {len(sims)} simulations")
    """
    base_path = Path(base_path) if isinstance(base_path, str) else base_path

    if not base_path.is_dir():
        return []

    try:
        sims = vvm.list_available_simulations(str(base_path))
        return sims
    except Exception as e:
        logger.error(f"Error listing simulations in {base_path}: {e}")
        return []


# =============================================================================
# Variable Group Scanning
# =============================================================================

def scan_variable_groups(sim_path: Path | str) -> Dict[str, List[str]]:
    """
    Scan a VVM simulation directory to identify available variable groups.

    This function examines NetCDF files in the archive/ subdirectory to
    determine which variable groups are available (e.g., 'C.Surface', 'L.Dynamic').

    Parameters
    ----------
    sim_path : Path or str
        Path to VVM simulation directory

    Returns
    -------
    dict
        Dictionary mapping group names to lists of variables.
        Format: {'File: <group_name>': [var1, var2, ...], ...}

    Examples
    --------
    >>> groups = scan_variable_groups('/data2/VVM/sim001/')
    >>> for group_name, variables in groups.items():
    ...     print(f"{group_name}: {len(variables)} variables")
    """
    sim_path = Path(sim_path) if isinstance(sim_path, str) else sim_path

    logger.info(f"Scanning simulation directory: {sim_path}")

    # Find all initial timestep files (ending in -000000.nc)
    archive_dir = sim_path / 'archive'
    files = sorted(glob(str(archive_dir / "*-000000.nc")))

    # Map group names to file paths
    group_map = {}
    for f in files:
        fname = os.path.basename(f)
        # Extract group name from filename (e.g., '.L.Thermodynamic-')
        match = re.search(r'\.([CL]\.[A-Za-z0-9]+)-', fname)
        if match:
            group = match.group(1)
            if group not in group_map:
                group_map[group] = f

    # Scan each group file for variables
    menu_dict = {}
    for group, fpath in group_map.items():
        try:
            with xr.open_dataset(fpath, chunks={}) as ds_meta:
                # Filter out coordinate variables
                vars_list = [
                    v for v in ds_meta.data_vars
                    if v not in ['xc', 'yc', 'zc', 'time', 'lon', 'lat', 'lev']
                ]
                if vars_list:
                    menu_dict[f"File: {group}"] = sorted(vars_list)
        except Exception as e:
            logger.warning(f"Could not read {fpath}: {e}")

    # Add diagnostic variables
    try:
        diag_vars = vvm.list_available_diagnostics()
        if diag_vars:
            menu_dict["Calc: Diagnostics"] = sorted(diag_vars)
    except Exception as e:
        logger.warning(f"Could not list diagnostics: {e}")

    # Enrich with derived variables (e.g., Topography)
    menu_dict = enrich_variable_groups(menu_dict)

    return menu_dict


def enrich_variable_groups(groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Add derived variables (e.g., terrain height) to variable groups.

    Parameters
    ----------
    groups : dict
        Original variable groups from scan_variable_groups()

    Returns
    -------
    dict
        Enriched variable groups including terrain height
    """
    groups = dict(groups)  # Create a copy
    groups["File: Topography"] = [TERRAIN_VAR_NAME]
    return groups


# =============================================================================
# Dataset Loading with Caching
# =============================================================================

def _create_time_selection(t_range: Tuple) -> vvm.TimeSelection:
    """
    Create a vvm_reader TimeSelection object.

    Parameters
    ----------
    t_range : tuple
        Time range specification:
        - (start, end): Index-based selection (default)
        - ('index', start, end): Explicitly index-based
        - ('time', start, end): Time-based selection

    Returns
    -------
    vvm.TimeSelection
        TimeSelection object for vvm_reader
    """
    # Check for explicit mode specification
    if len(t_range) == 3 and isinstance(t_range[0], str):
        mode = t_range[0]
        if mode == 'time':
            return vvm.TimeSelection(time_range=t_range[1:])
        elif mode == 'index':
            return vvm.TimeSelection(time_index_range=t_range[1:])

    # Default: index-based selection
    return vvm.TimeSelection(time_index_range=t_range)


def _create_vertical_selection(z_range: Tuple) -> Optional[vvm.VerticalSelection]:
    """
    Create a vvm_reader VerticalSelection object.

    Parameters
    ----------
    z_range : tuple or None
        Vertical range specification:
        - None: No vertical selection (for 2D surface variables)
        - (start, end): Height-based selection (default, in meters)
        - ('height', start, end): Explicitly height-based
        - ('index', start, end): Index-based selection (vertical level index)

    Returns
    -------
    vvm.VerticalSelection or None
        VerticalSelection object for vvm_reader, or None for 2D variables
    """
    # Handle None for 2D surface variables
    if z_range is None:
        return None
    
    # Check for explicit mode specification
    if len(z_range) == 3 and isinstance(z_range[0], str):
        mode = z_range[0]
        if mode == 'index':
            return vvm.VerticalSelection(index_range=z_range[1:])
        elif mode == 'height':
            return vvm.VerticalSelection(height_range=z_range[1:])

    # Default: height-based selection
    return vvm.VerticalSelection(height_range=z_range)


def _open_dataset(
    sim_path: Path | str,
    var_name: str,
    t_range: Tuple | Dict,
    z_range: Tuple | Dict,
    x_range: Tuple[int, int],
    y_range: Tuple[int, int]
) -> xr.Dataset:
    """
    Open a VVM dataset for a specific variable with given ranges.

    This is the internal, non-cached version. Use open_dataset() for cached access.

    Parameters
    ----------
    sim_path : Path or str
        Path to VVM simulation directory
    var_name : str
        Variable name to load
    t_range : tuple or dict
        Time range specification
    z_range : tuple or dict
        Vertical range specification
    x_range : tuple of int
        X (longitude) index range (start, end)
    y_range : tuple of int
        Y (latitude) index range (start, end)

    Returns
    -------
    xr.Dataset
        Loaded dataset (lazy-loaded with Dask chunks)
    """
    logger.debug(f"Opening dataset: {var_name} in {sim_path}")

    # Optimized chunking strategy:
    # - Column-integrated variables (cwv, lwp, iwp) need full vertical columns
    #   (lev: -1) for efficient integration without cross-chunk communication
    # - 3D variables benefit from lev: 1 for efficient slicing
    if var_name in ['cwv', 'lwp', 'iwp']:
        chunks = {'time': 1, 'lev': -1, 'lat': 128, 'lon': 128}
    else:
        chunks = {'time': 1, 'lev': 1, 'lat': -1, 'lon': -1}

    # Processing options for vvm_reader
    opts = vvm.ProcessingOptions(
        center_staggered=True,  # Center staggered grid variables
        center_suffix='',       # No suffix for centered variables
        chunks=chunks,
    )

    # Create selection objects
    time_sel = _create_time_selection(t_range)
    vert_sel = _create_vertical_selection(z_range)
    region = vvm.Region(x_range=x_range, y_range=y_range)

    # Protect HDF5/NetCDF read with lock (prevent HDF5 segfaults)
    with FILE_IO_LOCK:
        ds = vvm.open_vvm_dataset(
            str(sim_path),
            variables=[var_name],
            time_selection=time_sel,
            vertical_selection=vert_sel,
            region=region,
            processing_options=opts
        )

    return ds


@lru_cache(maxsize=DATASET_CACHE_SIZE)
def open_dataset(
    sim_path: str,  # Must be hashable for lru_cache
    var_name: str,
    t_range: Tuple,
    z_range: Tuple,
    x_range: Tuple[int, int],
    y_range: Tuple[int, int]
) -> xr.Dataset:
    """
    Open a VVM dataset with LRU caching.

    This function caches lazy-loaded xarray.Dataset objects at the file I/O level.
    It maintains up to DATASET_CACHE_SIZE (default: 10) datasets in memory.

    IMPORTANT: All parameters must be hashable for caching. Convert Path objects
    to strings and use tuples instead of lists.

    Parameters
    ----------
    sim_path : str
        Path to VVM simulation directory (must be string for caching)
    var_name : str
        Variable name to load
    t_range : tuple
        Time range specification (must be tuple for caching)
    z_range : tuple
        Vertical range specification (must be tuple for caching)
    x_range : tuple of int
        X (longitude) index range (start, end)
    y_range : tuple of int
        Y (latitude) index range (start, end)

    Returns
    -------
    xr.Dataset
        Cached lazy-loaded dataset

    Examples
    --------
    >>> ds = open_dataset(
    ...     '/data2/VVM/sim001/',
    ...     'qc',
    ...     t_range=(0, 100),
    ...     z_range=(0, 20000),
    ...     x_range=(0, 256),
    ...     y_range=(0, 256)
    ... )
    >>> print(ds.data_vars)
    """
    return _open_dataset(sim_path, var_name, t_range, z_range, x_range, y_range)


# =============================================================================
# Terrain Data Loading
# =============================================================================

# Terrain data cache: {sim_path: terrain_dataarray}
_terrain_cache: Dict[str, xr.DataArray] = {}


def get_terrain_data(
    sim_path: Path | str,
    x_range: Optional[Tuple[int, int]] = None,
    y_range: Optional[Tuple[int, int]] = None
) -> xr.DataArray:
    """
    Load terrain height data for a simulation.

    Terrain data is cached per simulation to avoid repeated I/O.
    If x_range and y_range are provided, returns a sliced version.

    Parameters
    ----------
    sim_path : Path or str
        Path to VVM simulation directory
    x_range : tuple of int, optional
        X (longitude) index range for slicing (start, end)
    y_range : tuple of int, optional
        Y (latitude) index range for slicing (start, end)

    Returns
    -------
    xr.DataArray
        Terrain height data with name set to TERRAIN_VAR_NAME

    Examples
    --------
    >>> terrain = get_terrain_data('/data2/VVM/sim001/')
    >>> print(terrain.shape)

    >>> terrain_sliced = get_terrain_data(
    ...     '/data2/VVM/sim001/',
    ...     x_range=(0, 128),
    ...     y_range=(0, 128)
    ... )
    """
    sim_path_str = str(sim_path)

    # Check cache
    terrain_da = _terrain_cache.get(sim_path_str)
    if terrain_da is None:
        # Load terrain data with thread lock
        with FILE_IO_LOCK:
            terrain_da = vvm.get_terrain_height(sim_path_str)
        _terrain_cache[sim_path_str] = terrain_da

    # Slice if ranges provided
    if x_range is not None and y_range is not None:
        try:
            x0, x1 = x_range
            y0, y1 = y_range

            # Ensure valid ranges
            if x1 <= x0:
                x1 = x0 + 1
            if y1 <= y0:
                y1 = y0 + 1

            # Identify dimension names (handle both 'lon'/'lat' and generic dims)
            lon_dim = 'lon' if 'lon' in terrain_da.dims else list(terrain_da.dims)[-1]
            lat_dim = 'lat' if 'lat' in terrain_da.dims else list(terrain_da.dims)[0]

            sliced = terrain_da.isel({lon_dim: slice(x0, x1), lat_dim: slice(y0, y1)})
        except Exception as e:
            logger.warning(f"Could not slice terrain data: {e}")
            sliced = terrain_da
    else:
        sliced = terrain_da

    # Set name for consistency
    sliced.name = TERRAIN_VAR_NAME

    return sliced


# =============================================================================
# Simulation Metadata Queries
# =============================================================================

def get_coordinate_info(sim_path: Path | str) -> Dict[str, Any]:
    """
    Get coordinate information for a simulation.

    Parameters
    ----------
    sim_path : Path or str
        Path to VVM simulation directory

    Returns
    -------
    dict
        Dictionary with keys: 'nx', 'ny', and other coordinate info
        from vvm_reader.get_coordinate_info()

    Examples
    --------
    >>> info = get_coordinate_info('/data2/VVM/sim001/')
    >>> print(f"Grid size: {info['nx']} x {info['ny']}")
    """
    with FILE_IO_LOCK:
        return vvm.get_coordinate_info(str(sim_path))


def get_vertical_info(sim_path: Path | str) -> Dict[str, Any]:
    """
    Get vertical coordinate information for a simulation.

    Parameters
    ----------
    sim_path : Path or str
        Path to VVM simulation directory

    Returns
    -------
    dict
        Dictionary with keys: 'height_range', 'nz', and other vertical info
        from vvm_reader.get_vertical_info()

    Examples
    --------
    >>> info = get_vertical_info('/data2/VVM/sim001/')
    >>> print(f"Height range: {info['height_range']}")
    """
    with FILE_IO_LOCK:
        return vvm.get_vertical_info(str(sim_path))


def get_terrain_info(sim_path: Path | str) -> Optional[Dict[str, Any]]:
    """
    Get terrain information for a simulation.

    Parameters
    ----------
    sim_path : Path or str
        Path to VVM simulation directory

    Returns
    -------
    dict or None
        Dictionary with keys: 'max_level', 'min_level', etc.
        Returns None if terrain info cannot be retrieved.

    Examples
    --------
    >>> info = get_terrain_info('/data2/VVM/sim001/')
    >>> if info:
    ...     is_flat = (info['max_level'] == info['min_level'] == 0)
    """
    try:
        with FILE_IO_LOCK:
            return vvm.get_terrain_info(str(sim_path))
    except Exception as e:
        logger.warning(f"Could not get terrain info for {sim_path}: {e}")
        return None


def scan_time_indices(sim_path: Path | str) -> List[int]:
    """
    Scan available time indices from NetCDF filenames in archive/.

    This examines the archive/ directory to find available
    time indices based on filename patterns (e.g., *-000000.nc, *-000120.nc).
    
    Returns a sorted list of actual available indices, not just (min, max),
    to handle simulations with missing time files correctly.

    Parameters
    ----------
    sim_path : Path or str
        Path to VVM simulation directory

    Returns
    -------
    list of int
        Sorted list of available time indices (unique values).
        Returns [0] as fallback if no files found.

    Examples
    --------
    >>> indices = scan_time_indices('/data2/VVM/sim001/')
    >>> print(f"Available: {len(indices)} time steps, range {indices[0]}-{indices[-1]}")
    """
    sim_path = Path(sim_path) if isinstance(sim_path, str) else sim_path

    archive_dir = sim_path / 'archive'
    nc_files = sorted(glob(str(archive_dir / "*.nc")))

    indices = set()  # Use set to avoid duplicates from different variable files
    for f in nc_files:
        # Extract 6-digit index from filename (e.g., '-000120.nc')
        match = re.search(r'-(\d{6})\.nc$', f)
        if match:
            indices.add(int(match.group(1)))

    if indices:
        return sorted(indices)
    else:
        # Fallback
        return [0]
