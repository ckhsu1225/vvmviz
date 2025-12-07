"""
VVM Data Processing Module

Provides high-level functions for extracting and processing VVM data,
including main variables, wind vectors, and contour overlays.
"""

import logging
from typing import Tuple, Optional, Dict, Any
import xarray as xr

from vvmviz.config import TERRAIN_VAR_NAME, FILE_IO_LOCK
from vvmviz.core.data_loader import open_dataset, get_terrain_data

logger = logging.getLogger(__name__)


# =============================================================================
# Data Array Extraction
# =============================================================================

def get_data_array(
    sim_path: str,
    var_name: str,
    t_range: Tuple,
    z_range: Tuple,
    x_range: Tuple[int, int],
    y_range: Tuple[int, int],
    use_cache: bool = True
) -> Optional[xr.DataArray]:
    """
    Get a DataArray for a specific variable with range selections.

    This is the main entry point for accessing VVM data. It handles both
    regular variables and special cases like terrain height.

    Parameters
    ----------
    sim_path : str
        Path to VVM simulation directory
    var_name : str
        Variable name to load
    t_range : tuple
        Time range specification
    z_range : tuple
        Vertical range specification
    x_range : tuple of int
        X (longitude) index range (start, end)
    y_range : tuple of int
        Y (latitude) index range (start, end)
    use_cache : bool, default=True
        Whether to use cached datasets

    Returns
    -------
    xr.DataArray or None
        Requested variable data array, or None if loading fails

    Examples
    --------
    >>> da = get_data_array(
    ...     '/data2/VVM/sim001/',
    ...     'qc',
    ...     t_range=(0, 100),
    ...     z_range=(0, 20000),
    ...     x_range=(0, 256),
    ...     y_range=(0, 256)
    ... )
    >>> print(da.shape)
    """
    # Validate ranges to prevent Dask ZeroDivisionError on empty slices
    if x_range[0] == x_range[1]:
        x_range = (x_range[0], x_range[1] + 1)
    if y_range[0] == y_range[1]:
        y_range = (y_range[0], y_range[1] + 1)

    # Special case: terrain height
    if var_name == TERRAIN_VAR_NAME:
        return get_terrain_data(sim_path, x_range, y_range)

    # Load dataset
    if use_cache:
        ds = open_dataset(sim_path, var_name, t_range, z_range, x_range, y_range)
    else:
        # Import the internal uncached function
        from vvmviz.core.data_loader import _open_dataset
        ds = _open_dataset(sim_path, var_name, t_range, z_range, x_range, y_range)

    # Extract DataArray from dataset
    if var_name in ds:
        return ds[var_name]

    # Handle suffix mismatch (e.g., 'u' might be stored as 'u_sfc')
    for v in ds.data_vars:
        if v.startswith(var_name):
            return ds[v]

    return None


# =============================================================================
# Wind Vector Processing
# =============================================================================

def get_wind_vectors(
    sim_path: str,
    t_range: Tuple,
    z_range: Tuple,
    x_range: Tuple[int, int],
    y_range: Tuple[int, int],
    use_surface: bool = False,
    use_cache: bool = True
) -> Optional[Tuple[xr.DataArray, xr.DataArray]]:
    """
    Load wind vector components (u, v) for a given selection.

    Supports both level-based winds and surface winds (composite of
    ocean and land surface levels).

    Parameters
    ----------
    sim_path : str
        Path to VVM simulation directory
    t_range : tuple
        Time range specification
    z_range : tuple
        Vertical range specification (ignored if use_surface=True)
    x_range : tuple of int
        X (longitude) index range (start, end)
    y_range : tuple of int
        Y (latitude) index range (start, end)
    use_surface : bool, default=False
        If True, compute composite surface wind (ocean level 1, land level 2)
    use_cache : bool, default=True
        Whether to use cached datasets

    Returns
    -------
    tuple of xr.DataArray or None
        (u_component, v_component) wind vectors, or None if loading fails

    Examples
    --------
    >>> u, v = get_wind_vectors(
    ...     '/data2/VVM/sim001/',
    ...     t_range=(0, 0),
    ...     z_range=(1000, 1000),
    ...     x_range=(0, 256),
    ...     y_range=(0, 256)
    ... )
    """
    try:
        if use_surface:
            # Optimized Surface Wind: composite of ocean (lev 1) and land (lev 2)
            terrain_da = get_terrain_data(sim_path, x_range, y_range)
            land_mask = terrain_da > 0

            # Extract single time index
            if isinstance(t_range, tuple) and len(t_range) == 2:
                t_idx = t_range[0]
            else:
                t_idx = 0

            # Load ocean surface wind (index level 1)
            with FILE_IO_LOCK:
                u_ocean = get_data_array(
                    sim_path, 'u',
                    t_range=(t_idx, t_idx),
                    z_range=('index', 1, 1),
                    x_range=x_range,
                    y_range=y_range,
                    use_cache=use_cache
                )
                v_ocean = get_data_array(
                    sim_path, 'v',
                    t_range=(t_idx, t_idx),
                    z_range=('index', 1, 1),
                    x_range=x_range,
                    y_range=y_range,
                    use_cache=use_cache
                )

                # Load land surface wind (index level 2)
                u_land = get_data_array(
                    sim_path, 'u',
                    t_range=(t_idx, t_idx),
                    z_range=('index', 2, 2),
                    x_range=x_range,
                    y_range=y_range,
                    use_cache=use_cache
                )
                v_land = get_data_array(
                    sim_path, 'v',
                    t_range=(t_idx, t_idx),
                    z_range=('index', 2, 2),
                    x_range=x_range,
                    y_range=y_range,
                    use_cache=use_cache
                )

            # Check all components loaded successfully
            if all(x is not None for x in [u_ocean, v_ocean, u_land, v_land]):
                
                # Squeeze dimensions first to ensure alignment on lat/lon
                u_ocean = squeeze_singleton_dims(u_ocean)
                u_land = squeeze_singleton_dims(u_land)
                v_ocean = squeeze_singleton_dims(v_ocean)
                v_land = squeeze_singleton_dims(v_land)
                
                # Align land_mask to wind data to avoid 'join=exact' mismatch
                # Wind data and terrain might be sliced slightly differently
                land_mask = land_mask.reindex_like(u_ocean, method='nearest')
                
                # Composite: use ocean wind over ocean, land wind over land
                u_comp = u_ocean.where(~land_mask, u_land)
                v_comp = v_ocean.where(~land_mask, v_land)

                return (u_comp, v_comp)
            else:
                return None

        else:
            # Level-based wind
            with FILE_IO_LOCK:
                u_da = get_data_array(
                    sim_path, 'u',
                    t_range=t_range,
                    z_range=z_range,
                    x_range=x_range,
                    y_range=y_range,
                    use_cache=use_cache
                )
                v_da = get_data_array(
                    sim_path, 'v',
                    t_range=t_range,
                    z_range=z_range,
                    x_range=x_range,
                    y_range=y_range,
                    use_cache=use_cache
                )

            if u_da is not None and v_da is not None:
                return (u_da, v_da)
            else:
                return None

    except Exception as e:
        logger.error(f"Error loading wind vectors: {e}")
        return None


# =============================================================================
# Contour Overlay Processing
# =============================================================================

def get_contour_data(
    sim_path: str,
    var_name: str,
    t_range: Tuple,
    z_range: Tuple,
    x_range: Tuple[int, int],
    y_range: Tuple[int, int],
    use_cache: bool = True
) -> Optional[xr.DataArray]:
    """
    Load data for contour overlay variable.

    This is a convenience wrapper around get_data_array() specifically
    for loading contour overlay data.

    Parameters
    ----------
    sim_path : str
        Path to VVM simulation directory
    var_name : str
        Contour variable name
    t_range : tuple
        Time range specification
    z_range : tuple
        Vertical range specification
    x_range : tuple of int
        X (longitude) index range (start, end)
    y_range : tuple of int
        Y (latitude) index range (start, end)
    use_cache : bool, default=True
        Whether to use cached datasets

    Returns
    -------
    xr.DataArray or None
        Contour variable data, or None if loading fails

    Examples
    --------
    >>> contour_da = get_contour_data(
    ...     '/data2/VVM/sim001/',
    ...     'w',
    ...     t_range=(0, 0),
    ...     z_range=(1000, 1000),
    ...     x_range=(0, 256),
    ...     y_range=(0, 256)
    ... )
    """
    try:
        da = get_data_array(
            sim_path, var_name,
            t_range=t_range,
            z_range=z_range,
            x_range=x_range,
            y_range=y_range,
            use_cache=use_cache
        )
        return da
    except Exception as e:
        logger.error(f"Error loading contour data for {var_name}: {e}")
        return None


# =============================================================================
# Multi-layer Frame Bundle Loading
# =============================================================================

def load_frame_bundle(
    sim_path: str,
    main_var: str,
    t_range: Tuple,
    z_range: Tuple,
    x_range: Tuple[int, int],
    y_range: Tuple[int, int],
    wind_enabled: bool = False,
    use_surface_wind: bool = False,
    contour_enabled: bool = False,
    contour_var: Optional[str] = None,
    use_cache: bool = True,
    compute: bool = False
) -> Dict[str, Any]:
    """
    Load a complete frame bundle including main variable, wind, and contour.

    This function loads all components needed for a single visualization frame:
    - Main variable
    - Wind vectors (optional)
    - Contour overlay (optional)

    Parameters
    ----------
    sim_path : str
        Path to VVM simulation directory
    main_var : str
        Main variable name
    t_range : tuple
        Time range specification
    z_range : tuple
        Vertical range specification
    x_range : tuple of int
        X (longitude) index range (start, end)
    y_range : tuple of int
        Y (latitude) index range (start, end)
    wind_enabled : bool, default=False
        Whether to load wind vectors
    use_surface_wind : bool, default=False
        Whether to use composite surface wind
    contour_enabled : bool, default=False
        Whether to load contour overlay
    contour_var : str, optional
        Contour variable name (required if contour_enabled=True)
    use_cache : bool, default=True
        Whether to use cached datasets
    compute : bool, default=False
        Whether to compute (load into memory) Dask arrays immediately

    Returns
    -------
    dict
        Bundle dictionary with keys:
        - 't_range': Time range used
        - 'z_range': Vertical range used
        - 'main_var': Main variable name
        - 'main': Main variable DataArray (optional)
        - 'wind': Tuple of (u, v) DataArrays (optional)
        - 'contour': Contour variable DataArray (optional)

    Examples
    --------
    >>> bundle = load_frame_bundle(
    ...     '/data2/VVM/sim001/',
    ...     main_var='qc',
    ...     t_range=(0, 0),
    ...     z_range=(1000, 1000),
    ...     x_range=(0, 256),
    ...     y_range=(0, 256),
    ...     wind_enabled=True,
    ...     contour_enabled=True,
    ...     contour_var='w'
    ... )
    >>> print('main' in bundle, 'wind' in bundle, 'contour' in bundle)
    """
    result = {
        't_range': t_range,
        'z_range': z_range,
        'main_var': main_var
    }

    # 1. Load Main Variable
    with FILE_IO_LOCK:
        da = get_data_array(
            sim_path, main_var,
            t_range=t_range,
            z_range=z_range,
            x_range=x_range,
            y_range=y_range,
            use_cache=use_cache
        )

    if da is not None:
        result['main'] = da.compute() if compute else da

    # 2. Load Wind (if enabled)
    if wind_enabled:
        wind_data = get_wind_vectors(
            sim_path,
            t_range=t_range,
            z_range=z_range,
            x_range=x_range,
            y_range=y_range,
            use_surface=use_surface_wind,
            use_cache=use_cache
        )

        if wind_data is not None:
            u_da, v_da = wind_data
            if compute:
                result['wind'] = (u_da.compute(), v_da.compute())
            else:
                result['wind'] = (u_da, v_da)

    # 3. Load Contour (if enabled)
    if contour_enabled and contour_var:
        contour_da = get_contour_data(
            sim_path, contour_var,
            t_range=t_range,
            z_range=z_range,
            x_range=x_range,
            y_range=y_range,
            use_cache=use_cache
        )

        if contour_da is not None:
            result['contour'] = contour_da.compute() if compute else contour_da

    return result


# =============================================================================
# Dimension Processing Utilities
# =============================================================================

def squeeze_singleton_dims(da: xr.DataArray) -> xr.DataArray:
    """
    Remove singleton dimensions from a DataArray.

    Useful for cleaning up DataArrays after selecting a single time step
    or vertical level.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray

    Returns
    -------
    xr.DataArray
        DataArray with singleton dimensions removed

    Examples
    --------
    >>> da = xr.DataArray(np.ones((1, 1, 256, 256)),
    ...                   dims=['time', 'lev', 'lat', 'lon'])
    >>> da_clean = squeeze_singleton_dims(da)
    >>> print(da_clean.dims)
    ('lat', 'lon')
    """
    # Identify dimensions with size 1
    singleton_dims = [dim for dim in da.dims if da.sizes[dim] == 1]

    # Squeeze them out
    if singleton_dims:
        da = da.squeeze(singleton_dims)

    return da


def select_single_time_level(
    da: xr.DataArray,
    time_idx: Optional[int] = None,
    lev_idx: Optional[int] = None
) -> xr.DataArray:
    """
    Select a single time step and/or vertical level from a DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray
    time_idx : int, optional
        Time index to select (if None, not selected)
    lev_idx : int, optional
        Level index to select (if None, not selected)

    Returns
    -------
    xr.DataArray
        DataArray with selected time and/or level

    Examples
    --------
    >>> da = xr.DataArray(np.random.rand(10, 50, 256, 256),
    ...                   dims=['time', 'lev', 'lat', 'lon'])
    >>> da_slice = select_single_time_level(da, time_idx=5, lev_idx=10)
    >>> print(da_slice.shape)
    (256, 256)
    """
    if time_idx is not None and 'time' in da.dims:
        da = da.isel(time=time_idx)

    if lev_idx is not None and 'lev' in da.dims:
        da = da.isel(lev=lev_idx)

    return da
