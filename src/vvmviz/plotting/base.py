"""
Core Plotting Module

This module provides the core plotting functionality for VVMViz:
- Main image rendering with HoloViews and Datashader
- Color limit handling (auto, locked, symmetric)
- Multi-layer composition (main image + overlays)
- Zoom/pan state management
"""

import logging
from typing import Optional, Tuple, List, Any

import xarray as xr
import holoviews as hv
import holoviews.operation.datashader as hd

from vvmviz.plotting.colormaps import resolve_colormap

logger = logging.getLogger(__name__)


# =============================================================================
# Color Limit Handling
# =============================================================================

def calculate_color_limits(
    da: xr.DataArray,
    lock_clim: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    symmetric: bool = False
) -> Tuple[float, float]:
    """
    Calculate color limits for a DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Data array to analyze
    lock_clim : bool, default=False
        If True, use provided vmin/vmax; if False, compute from data
    vmin : float, optional
        Minimum value (used when lock_clim=True)
    vmax : float, optional
        Maximum value (used when lock_clim=True)
    symmetric : bool, default=False
        If True, use symmetric limits around zero

    Returns
    -------
    Tuple[float, float]
        (vmin, vmax) color limits

    Examples
    --------
    >>> data = xr.DataArray(np.random.randn(100, 100))
    >>> vmin, vmax = calculate_color_limits(data, symmetric=True)
    >>> vmin == -vmax  # Symmetric
    True
    """
    if lock_clim and vmin is not None and vmax is not None:
        # Use locked values
        return (float(vmin), float(vmax))

    # Compute from data
    current_min = float(da.min().values)
    current_max = float(da.max().values)

    if symmetric:
        abs_max = max(abs(current_min), abs(current_max))
        return (-abs_max, abs_max)
    else:
        return (current_min, current_max)


# =============================================================================
# Main Image Creation
# =============================================================================

def create_image(
    da: xr.DataArray,
    cmap: Any,
    clim: Tuple[float, float],
    scale: str = 'Linear',
    title: str = '',
    hover_dims: Optional[List[str]] = None,
    frame_height: int = 500,
    alpha: float = 0.9,
    dynamic: bool = True
) -> hv.DynamicMap:
    """
    Create main image visualization with Datashader rasterization.

    This function:
    1. Converts DataArray to HoloViews Dataset
    2. Creates Image element
    3. Applies Datashader rasterization for performance
    4. Configures colormap, limits, and interactive tools

    Parameters
    ----------
    da : xr.DataArray
        2D data array to visualize (must have 'lon', 'lat' dimensions or similar)
    cmap : colormap
        Colormap object (from resolve_colormap)
    clim : tuple of float
        (vmin, vmax) color limits
    scale : str, default='Linear'
        Color scaling: 'Linear' or 'Log'
    title : str, default=''
        Plot title
    hover_dims : list of str, optional
        Additional dimensions to show in hover tooltip
    frame_height : int, default=500
        Plot height in pixels
    alpha : float, default=0.9
        Image transparency (0-1)
    dynamic : bool, default=True
        Use dynamic rasterization (recomputes on zoom)

    Returns
    -------
    hv.DynamicMap
        Rasterized image ready for display

    Examples
    --------
    >>> from vvmviz.plotting.colormaps import resolve_colormap
    >>> data = xr.DataArray(
    ...     np.random.rand(256, 256),
    ...     dims=['lat', 'lon'],
    ...     coords={'lat': np.linspace(0, 10, 256), 'lon': np.linspace(0, 10, 256)}
    ... )
    >>> cmap = resolve_colormap('viridis')
    >>> img = create_image(data, cmap, clim=(0, 1), title='Test')
    """
    try:
        # Determine dimension names
        # Use coordinates if available, else infer from dims
        # For Dataset, use coordinates common to data vars or just use dims of the main var
        if isinstance(da, xr.Dataset):
            # Attempt to find main variable
            var_name = da.attrs.get('main_var')
            if not var_name:
                # Fallback: take the first data variable
                var_name = list(da.data_vars)[0]
            
            # Use dims of the main variable
            main_da = da[var_name]
            lon_dim = 'lon' if 'lon' in main_da.dims else main_da.dims[-1]
            lat_dim = 'lat' if 'lat' in main_da.dims else main_da.dims[0]
        else:
            # DataArray
            var_name = da.name if da.name else 'data'
            lon_dim = 'lon' if 'lon' in da.dims else da.dims[-1]
            lat_dim = 'lat' if 'lat' in da.dims else da.dims[0]

        # Create dimension objects with nice labels
        lon_label = 'longitude'
        lat_label = 'latitude'
        
        # Try to get units or long_name if available
        if lon_dim in da.coords:
            units = da.coords[lon_dim].attrs.get('units', '')
            if units:
                lon_label += f' ({units})'
        
        if lat_dim in da.coords:
            units = da.coords[lat_dim].attrs.get('units', '')
            if units:
                lat_label += f' ({units})'

        kdims = [
            hv.Dimension(lon_dim, label=lon_label),
            hv.Dimension(lat_dim, label=lat_label)
        ]

        # Create dimension for the main variable with label
        if isinstance(da, xr.Dataset):
            da_attrs = da[var_name].attrs
        else:
            da_attrs = da.attrs
            
        var_label = var_name
        long_name = da_attrs.get('long_name', '')
        units = da_attrs.get('units', '')
        
        if long_name:
            var_label = long_name
        if units and units != 'N/A':
            var_label += f' ({units})'

        # Prepare value dimensions for hover
        # Use hv.Dimension for the main variable to enforce the label
        main_vdim = hv.Dimension(var_name, label=var_label)
        vdims = [main_vdim]
        
        if hover_dims:
            vdims.extend(hover_dims)

        # Convert to HoloViews Dataset
        if isinstance(da, xr.Dataset):
            ds = hv.Dataset(da, kdims=kdims, vdims=vdims)
        else:
            ds = hv.Dataset(da, kdims=kdims, vdims=vdims)

        # Create raw image element
        raw_img = ds.to(hv.Image, kdims=kdims, vdims=vdims)

        # Apply Datashader rasterization
        img = hd.rasterize(raw_img, dynamic=dynamic).opts(
            cmap=cmap,
            clim=clim,
            colorbar=True,
            alpha=alpha,
            cnorm='log' if scale == 'Log' else 'linear',
            frame_height=frame_height,
            aspect='equal',
            title=title,
            tools=['hover'],
            active_tools=['pan', 'wheel_zoom']
        )

        logger.debug(f"Created image for {var_name}, clim={clim}, scale={scale}")
        return img

    except Exception as e:
        logger.error(f"Error creating image: {e}")
        raise


# =============================================================================
# Multi-Layer Composition
# =============================================================================

def compose_plot(
    base_image: hv.DynamicMap,
    overlays: Optional[List[Optional[hv.Element]]] = None
) -> hv.Element:
    """
    Compose final plot by overlaying elements on base image.

    Overlays are stacked in the order provided (first overlay is bottom-most).
    None values in overlays list are skipped.

    Parameters
    ----------
    base_image : hv.DynamicMap
        Base rasterized image
    overlays : list of hv.Element, optional
        List of overlay elements to stack on top.
        Common order: [boundaries, contours, wind_vectors]

    Returns
    -------
    hv.Element
        Composed plot with all layers

    Examples
    --------
    >>> base = create_image(data, cmap, clim=(0, 1))
    >>> wind = create_wind_vectors(u, v)
    >>> contour = create_contour_overlay(data_c)
    >>> final = compose_plot(base, overlays=[contour, wind])
    """
    final_plot = base_image

    if overlays:
        for overlay in overlays:
            if overlay is not None:
                final_plot = final_plot * overlay
                logger.debug(f"Added overlay: {type(overlay).__name__}")

    return final_plot


# =============================================================================
# Zoom/Pan Range Management
# =============================================================================

def apply_ranges(
    plot_obj: hv.Element,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None
) -> hv.Element:
    """
    Apply x/y range limits to a plot object.

    Used to restore zoom/pan state when creating new plots.

    Parameters
    ----------
    plot_obj : hv.Element
        Plot object to apply ranges to
    x_range : tuple of float, optional
        (xmin, xmax) range
    y_range : tuple of float, optional
        (ymin, ymax) range

    Returns
    -------
    hv.Element
        Plot with ranges applied

    Examples
    --------
    >>> plot = create_image(data, cmap, clim=(0, 1))
    >>> zoomed_plot = apply_ranges(plot, x_range=(5, 8), y_range=(2, 6))
    """
    if x_range and all(v is not None for v in x_range):
        plot_obj = plot_obj.opts(xlim=x_range)
        logger.debug(f"Applied x_range: {x_range}")

    if y_range and all(v is not None for v in y_range):
        plot_obj = plot_obj.opts(ylim=y_range)
        logger.debug(f"Applied y_range: {y_range}")

    return plot_obj


# =============================================================================
# High-Level Plotting Function
# =============================================================================

def create_main_plot(
    da: xr.DataArray,
    cmap_name: str = 'viridis',
    reverse_cmap: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    lock_clim: bool = False,
    symmetric_clim: bool = False,
    scale: str = 'Linear',
    title: str = '',
    hover_dims: Optional[List[str]] = None,
    overlays: Optional[List[Optional[hv.Element]]] = None,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    frame_height: int = 500,
    alpha: float = 0.9
) -> hv.Element:
    """
    High-level function to create complete visualization.

    This function orchestrates the full plotting pipeline:
    1. Resolve colormap
    2. Calculate color limits
    3. Create base image with rasterization
    4. Add overlays
    5. Apply zoom/pan ranges

    Parameters
    ----------
    da : xr.DataArray
        2D data array to visualize
    cmap_name : str, default='viridis'
        Colormap name
    reverse_cmap : bool, default=False
        Reverse the colormap
    vmin : float, optional
        Minimum color value (if lock_clim=True)
    vmax : float, optional
        Maximum color value (if lock_clim=True)
    lock_clim : bool, default=False
        Lock color limits to vmin/vmax
    symmetric_clim : bool, default=False
        Use symmetric color limits around zero
    scale : str, default='Linear'
        'Linear' or 'Log' color scaling
    title : str, default=''
        Plot title
    hover_dims : list of str, optional
        Additional dimensions for hover tooltip
    overlays : list of hv.Element, optional
        Overlay elements (boundaries, contours, vectors)
    x_range : tuple of float, optional
        Zoom x range
    y_range : tuple of float, optional
        Zoom y range
    frame_height : int, default=500
        Plot height in pixels
    alpha : float, default=0.9
        Image transparency

    Returns
    -------
    hv.Element
        Complete visualization with all layers

    Examples
    --------
    Basic plot without overlays:

    >>> data = xr.DataArray(
    ...     np.random.randn(256, 256),
    ...     dims=['lat', 'lon'],
    ...     coords={'lat': np.linspace(0, 10, 256), 'lon': np.linspace(0, 10, 256)},
    ...     name='temperature'
    ... )
    >>> plot = create_main_plot(
    ...     data,
    ...     cmap_name='RdBu_r',
    ...     symmetric_clim=True,
    ...     title='Temperature Anomaly'
    ... )

    With overlays (boundaries and contours):

    >>> from vvmviz.plotting import get_county_boundaries, create_contour_overlay
    >>> # Create boundary and contour overlays
    >>> county = get_county_boundaries(color='black', line_width=1.0)
    >>> contours = create_contour_overlay(data, num_levels=5, cmap_name='gray')
    >>> # Compose plot with overlays
    >>> plot = create_main_plot(
    ...     data,
    ...     cmap_name='viridis',
    ...     overlays=[county, contours],
    ...     title='Temperature with Boundaries'
    ... )
    """
    # Step 1: Resolve colormap
    cmap = resolve_colormap(cmap_name, reverse=reverse_cmap)

    # Identify main variable for limits calculation
    if isinstance(da, xr.Dataset):
        var_name = da.attrs.get('main_var')
        if not var_name:
            var_name = list(da.data_vars)[0]
        main_da_for_limits = da[var_name]
    else:
        main_da_for_limits = da

    # Step 2: Calculate color limits
    clim = calculate_color_limits(
        main_da_for_limits,
        lock_clim=lock_clim,
        vmin=vmin,
        vmax=vmax,
        symmetric=symmetric_clim
    )

    # Step 3: Create base image
    base_image = create_image(
        da,
        cmap=cmap,
        clim=clim,
        scale=scale,
        title=title,
        hover_dims=hover_dims,
        frame_height=frame_height,
        alpha=alpha,
        dynamic=True
    )

    # Step 4: Compose with overlays
    final_plot = compose_plot(base_image, overlays)

    # Step 5: Determine and Apply Ranges
    # If ranges are not provided, default to the data bounds to prevent
    # overlays (like country boundaries) from expanding the view.
    
    # Identify main variable/dimensions to get bounds
    if isinstance(da, xr.Dataset):
        var_name = da.attrs.get('main_var')
        if not var_name:
            var_name = list(da.data_vars)[0]
        main_da_dims = da[var_name]
    else:
        main_da_dims = da

    lon_dim = 'lon' if 'lon' in main_da_dims.dims else main_da_dims.dims[-1]
    lat_dim = 'lat' if 'lat' in main_da_dims.dims else main_da_dims.dims[0]

    # Calculate data bounds if not provided
    if not x_range:
        try:
            x_min = float(main_da_dims[lon_dim].min())
            x_max = float(main_da_dims[lon_dim].max())
            x_range = (x_min, x_max)
        except Exception as e:
            logger.warning(f"Could not calculate x bounds: {e}")

    if not y_range:
        try:
            y_min = float(main_da_dims[lat_dim].min())
            y_max = float(main_da_dims[lat_dim].max())
            y_range = (y_min, y_max)
        except Exception as e:
            logger.warning(f"Could not calculate y bounds: {e}")

    # Apply ranges to the final plot
    final_plot = apply_ranges(final_plot, x_range, y_range)

    logger.info(f"Created main plot: {title or da.name}")
    return final_plot
