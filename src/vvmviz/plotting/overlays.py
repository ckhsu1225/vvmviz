"""
Overlay Plotting Module

This module provides overlay rendering functionality for VVMViz:
- Wind vector field overlays with dynamic density adjustment
- Contour line overlays with interactive legend
- Shapefile boundary overlays (county, town, custom)
"""

import logging
from typing import Optional, Tuple

import numpy as np
import xarray as xr
import holoviews as hv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import hvplot.xarray  # noqa: F401 - Required for xr.DataArray.hvplot
except ImportError:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Wind Vector Overlay
# =============================================================================

def create_wind_vectors(
    u_da: xr.DataArray,
    v_da: xr.DataArray,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    arrow_density: int = 20,
    arrow_scale: float = 1.0,
    color: str = 'black',
    magnitude: Optional[xr.DataArray] = None,
    angle_rad: Optional[xr.DataArray] = None
) -> Optional[hv.VectorField]:
    """
    Create wind vector field overlay from u and v components.

    Parameters
    ----------
    u_da : xr.DataArray
        Eastward wind component (2D: lat, lon)
    v_da : xr.DataArray
        Northward wind component (2D: lat, lon)
    x_range : tuple of float, optional
        (xmin, xmax) range to display
    y_range : tuple of float, optional
        (ymin, ymax) range to display
    arrow_density : int, default=20
        Target number of arrows along each dimension
    arrow_scale : float, default=1.0
        Scale factor for arrow size
    color : str, default='black'
        Arrow color
    magnitude : xr.DataArray, optional
        Pre-calculated wind speed. If None, computed from u, v.
    angle_rad : xr.DataArray, optional
        Pre-calculated wind angle in radians. If None, computed from u, v.

    Returns
    -------
    hv.VectorField or None
    """
    if u_da is None or v_da is None:
        logger.warning("Cannot create wind vectors: u or v data is None")
        return None

    try:
        # Determine dimension names
        lon_dim = 'lon' if 'lon' in u_da.dims else u_da.dims[-1]
        lat_dim = 'lat' if 'lat' in u_da.dims else u_da.dims[0]

        # Subset to viewing range if specified
        u_view = u_da
        v_view = v_da
        mag_view = magnitude
        ang_view = angle_rad

        if x_range is not None:
            lon_vals = u_da[lon_dim].values
            mask_lon = (lon_vals >= x_range[0]) & (lon_vals <= x_range[1])
            u_view = u_view.isel({lon_dim: mask_lon})
            v_view = v_view.isel({lon_dim: mask_lon})
            if mag_view is not None: mag_view = mag_view.isel({lon_dim: mask_lon})
            if ang_view is not None: ang_view = ang_view.isel({lon_dim: mask_lon})

        if y_range is not None:
            lat_vals = u_da[lat_dim].values
            mask_lat = (lat_vals >= y_range[0]) & (lat_vals <= y_range[1])
            u_view = u_view.isel({lat_dim: mask_lat})
            v_view = v_view.isel({lat_dim: mask_lat})
            if mag_view is not None: mag_view = mag_view.isel({lat_dim: mask_lat})
            if ang_view is not None: ang_view = ang_view.isel({lat_dim: mask_lat})

        # Compute downsampling skip factors
        skip_x = max(1, u_view.sizes.get(lon_dim, 1) // arrow_density)
        skip_y = max(1, u_view.sizes.get(lat_dim, 1) // arrow_density)

        # Downsample Coordinates
        u_down = u_view.isel({
            lon_dim: slice(None, None, skip_x),
            lat_dim: slice(None, None, skip_y)
        })
        
        # Compute or Downsample Magnitude/Angle
        if mag_view is not None:
            mag = mag_view.isel({
                lon_dim: slice(None, None, skip_x),
                lat_dim: slice(None, None, skip_y)
            })
        else:
            # Need u, v downsampled to compute
            v_down = v_view.isel({
                lon_dim: slice(None, None, skip_x),
                lat_dim: slice(None, None, skip_y)
            })
            mag = np.sqrt(u_down**2 + v_down**2)

        if ang_view is not None:
            angle = ang_view.isel({
                lon_dim: slice(None, None, skip_x),
                lat_dim: slice(None, None, skip_y)
            })
        else:
            # Re-use v_down if computed above, else compute
            if 'v_down' not in locals():
                v_down = v_view.isel({
                    lon_dim: slice(None, None, skip_x),
                    lat_dim: slice(None, None, skip_y)
                })
            angle = np.arctan2(v_down, u_down)

        # Extract coordinate values
        lons = u_down[lon_dim].values
        lats = u_down[lat_dim].values

        # Prepare columnar data for HoloViews
        xx, yy = np.meshgrid(lons, lats)
        x_flat = xx.flatten()
        y_flat = yy.flatten()
        angle_flat = angle.values.flatten()
        mag_flat = mag.values.flatten()

        # Filter out NaN values
        valid = ~np.isnan(mag_flat)
        if not np.any(valid):
            logger.warning("No valid wind vectors after filtering NaNs")
            return None

        vector_data = (
            x_flat[valid],
            y_flat[valid],
            angle_flat[valid],
            mag_flat[valid]
        )

        # Create VectorField
        vector_plot = hv.VectorField(
            vector_data,
            kdims=['lon', 'lat'],
            vdims=['Angle', 'Magnitude']
        ).opts(
            magnitude='Magnitude',
            scale=arrow_scale,
            color=color,
            tools=[],
            active_tools=[]
        )

        logger.debug(f"Created wind vector field with {np.sum(valid)} arrows")
        return vector_plot

    except Exception as e:
        logger.error(f"Error creating wind vectors: {e}")
        return None


# =============================================================================
# Contour Overlay
# =============================================================================

def create_contour_overlay(
    da: xr.DataArray,
    num_levels: int = 10,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_name: str = 'jet',
    line_width: float = 1.0
) -> Optional[hv.Overlay]:
    """
    Create contour line overlay with interactive legend.

    Each contour level is created as a separate layer to enable
    click-to-mute functionality in the legend.

    Parameters
    ----------
    da : xr.DataArray
        2D data array (lat, lon) to contour
    num_levels : int, default=10
        Number of contour levels to generate
    vmin : float, optional
        Minimum contour value. If None, uses data minimum
    vmax : float, optional
        Maximum contour value. If None, uses data maximum
    cmap_name : str, default='jet'
        Colormap name for contour coloring
    line_width : float, default=1.0
        Line width for contours

    Returns
    -------
    hv.Overlay or None
        Overlay of contour layers with click-to-mute policy, or None if failed

    Examples
    --------
    >>> data = xr.DataArray(
    ...     np.random.rand(100, 100),
    ...     dims=['lat', 'lon'],
    ...     coords={'lat': range(100), 'lon': range(100)}
    ... )
    >>> contours = create_contour_overlay(data, num_levels=5)
    """
    if da is None:
        logger.warning("Cannot create contours: data is None")
        return None

    try:
        # Ensure 2D data
        if da.ndim != 2:
            logger.error(f"Contour data must be 2D, got {da.ndim}D")
            return None

        # Determine dimension names
        lon_dim = 'lon' if 'lon' in da.dims else da.dims[-1]
        lat_dim = 'lat' if 'lat' in da.dims else da.dims[0]

        # Calculate value range
        data_min = float(da.min().values)
        data_max = float(da.max().values)

        if vmin is None:
            vmin = data_min
        if vmax is None:
            vmax = data_max

        # Generate contour levels
        if vmin == vmax:
            levels = [vmin]
            logger.debug(f"Single contour level: {vmin}")
        else:
            # Generate num_levels+2 points, then exclude endpoints to get num_levels
            levels = np.linspace(vmin, vmax, num_levels + 2)[1:-1]
            logger.debug(f"Generated {len(levels)} contour levels from {vmin} to {vmax}")

        if len(levels) == 0:
            logger.warning("No contour levels generated")
            return None

        # Get colormap
        try:
            cmap = plt.get_cmap(cmap_name)
        except Exception as e:
            logger.warning(f"Failed to load colormap {cmap_name}, using 'viridis': {e}")
            cmap = plt.get_cmap('viridis')

        # Normalize colors to levels range
        norm = mcolors.Normalize(vmin=min(levels), vmax=max(levels))

        # Create individual contour layers (one per level for interactive legend)
        contour_layers = []
        for lvl in levels:
            # Get color for this level
            val = norm(lvl)
            color = mcolors.to_hex(cmap(val))

            # Format label based on magnitude
            if abs(lvl) < 0.01 and abs(lvl) > 0:
                label = f"{lvl:.2e}"
            elif abs(lvl) < 1:
                label = f"{lvl:.3f}"
            else:
                label = f"{lvl:.1f}"

            # Create contour layer for this level
            try:
                layer = da.hvplot.contour(
                    x=lon_dim,
                    y=lat_dim,
                    levels=[lvl],
                    cmap=[color],
                    line_width=line_width,
                    hover=False,
                    dynamic=False,
                    rasterize=False,
                    framewise=False,
                    aspect='equal',
                    colorbar=False,
                    label=label,
                    legend=True
                ).opts(
                    color=color,
                    muted_alpha=0.1  # Faded when muted via legend click
                )
                contour_layers.append(layer)
            except Exception as e:
                logger.warning(f"Failed to create contour layer for level {lvl}: {e}")
                continue

        if not contour_layers:
            logger.warning("No contour layers created")
            return None

        # Combine into overlay with click-to-mute
        contour_overlay = hv.Overlay(contour_layers).opts(click_policy='mute')
        logger.debug(f"Created contour overlay with {len(contour_layers)} levels")

        return contour_overlay

    except Exception as e:
        logger.error(f"Error creating contour overlay: {e}")
        return None


# =============================================================================
# Shapefile Boundary Overlay
# =============================================================================

def get_county_boundaries(color: str = 'black', line_width: float = 1.0) -> hv.Path:
    """
    Get Taiwan county boundary overlay.

    Convenience function that loads county boundaries from the default path
    specified in config, optimized for use as map overlays.

    Parameters
    ----------
    color : str, optional
        Line color (default: 'black')
    line_width : float, optional
        Line width (default: 1.0)

    Returns
    -------
    hv.Path
        County boundary paths with tools disabled for overlay use

    Examples
    --------
    >>> from vvmviz.plotting import get_county_boundaries
    >>> county = get_county_boundaries(color='black', line_width=1.0)
    >>> plot = base_map * county  # Overlay on base map
    """
    from vvmviz.config import TWCOUNTY_SHP_PATH
    from vvmviz.utils.shapefile import load_boundary_paths

    cache_key = f"county_{color}_{line_width}"
    paths = load_boundary_paths(TWCOUNTY_SHP_PATH, cache_key, color, line_width)
    # Remove hover tool for overlay use
    return paths.opts(tools=[], active_tools=[])


def get_town_boundaries(color: str = 'gray', line_width: float = 0.5) -> hv.Path:
    """
    Get Taiwan town boundary overlay.

    Convenience function that loads town boundaries from the default path
    specified in config, optimized for use as map overlays.

    Parameters
    ----------
    color : str, optional
        Line color (default: 'gray')
    line_width : float, optional
        Line width (default: 0.5)

    Returns
    -------
    hv.Path
        Town boundary paths with tools disabled for overlay use

    Examples
    --------
    >>> from vvmviz.plotting import get_town_boundaries
    >>> town = get_town_boundaries(color='gray', line_width=0.5)
    >>> plot = base_map * town  # Overlay on base map
    """
    from vvmviz.config import TWTOWN_SHP_PATH
    from vvmviz.utils.shapefile import load_boundary_paths

    cache_key = f"town_{color}_{line_width}"
    paths = load_boundary_paths(TWTOWN_SHP_PATH, cache_key, color, line_width)
    # Remove hover tool for overlay use
    return paths.opts(tools=[], active_tools=[])
