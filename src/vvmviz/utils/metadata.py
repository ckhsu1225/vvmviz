"""
Metadata Formatting Utilities

Functions for formatting time values, building metadata displays,
and extracting variable information.
"""

import numpy as np
import xarray as xr
from typing import Dict, Any


def format_time_value(real_t: Any, units: str = "") -> str:
    """
    Format time value for display.

    Parameters
    ----------
    real_t : Any
        Time value (can be datetime64, numeric, or other type)
    units : str, optional
        Time units (e.g., "hours since 2024-01-01")

    Returns
    -------
    str
        Formatted time string
    """
    # Handle datetime64
    if isinstance(real_t, (np.datetime64, np.ndarray)):
        try:
            # Convert to pandas timestamp for easier formatting
            import pandas as pd
            ts = pd.Timestamp(real_t)
            return ts.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            return str(real_t)

    # Handle numeric types
    if isinstance(real_t, (int, float, np.integer, np.floating)):
        # If units provided, include them
        if units:
            return f"{real_t} ({units})"
        else:
            return f"{real_t}"

    # Fallback: convert to string
    return str(real_t)


def extract_metadata_from_dataarray(da: xr.DataArray) -> Dict[str, str]:
    """
    Extract metadata from an xarray DataArray.

    This reads metadata from DataArray attributes as provided by vvm_reader.

    Parameters
    ----------
    da : xr.DataArray
        DataArray to extract metadata from

    Returns
    -------
    dict
        Dictionary with 'name', 'long_name', and 'units' keys

    Examples
    --------
    >>> da = xr.DataArray([1, 2, 3], name='qc', attrs={'long_name': 'Cloud Water', 'units': 'kg/kg'})
    >>> meta = extract_metadata_from_dataarray(da)
    >>> print(meta['long_name'])
    Cloud Water
    """
    var_name = da.name if da.name else 'unknown'
    long_name = da.attrs.get('long_name', var_name)
    units = da.attrs.get('units', 'N/A')

    return {
        'name': var_name,
        'long_name': long_name,
        'units': units
    }


def build_metadata_markdown(
    da: xr.DataArray,
    contour_da: xr.DataArray | None = None
) -> str:
    """
    Build a formatted markdown string with DataArray metadata.

    This extracts metadata directly from the DataArray attributes
    as provided by vvm_reader.

    Parameters
    ----------
    da : xr.DataArray
        Main variable DataArray
    contour_da : xr.DataArray, optional
        Contour overlay variable DataArray (if enabled)

    Returns
    -------
    str
        Markdown-formatted metadata string
    """
    lines = ["### Variable Metadata\n\n"]

    # Main variable metadata
    var_meta = extract_metadata_from_dataarray(da)

    lines.append(f"- **Variable**: `{var_meta['name']}`\n")
    lines.append(f"- **Long Name**: {var_meta['long_name']}\n")
    lines.append(f"- **Units**: {var_meta['units']}\n")

    # Dimensions
    dims_str = " × ".join([f"{dim} ({da.sizes[dim]})" for dim in da.dims])
    lines.append(f"- **Dimensions**: {dims_str}\n")

    # Data range (compute only if data is small enough)
    try:
        if da.size < 1e6:  # Only compute for smaller arrays
            data_min = float(da.min().values)
            data_max = float(da.max().values)
            lines.append(f"- **Data Range**: [{data_min:.3e}, {data_max:.3e}]\n")
    except Exception:
        pass

    # Contour overlay metadata (if provided)
    if contour_da is not None:
        contour_meta = extract_metadata_from_dataarray(contour_da)
        lines.append("### Contour Overlay\n\n")
        lines.append(f"- **Variable**: `{contour_meta['name']}`\n")
        lines.append(f"- **Long Name**: {contour_meta['long_name']}\n")
        lines.append(f"- **Units**: {contour_meta['units']}\n")

    # Coordinates info
    lines.append("\n---\n")
    lines.append("\n### Coordinates\n")

    # Sort coordinates: preferred order first, then others
    preferred_order = ['time', 'lev', 'lat', 'lon', 'yc', 'xc']
    sorted_coords = sorted(
        list(da.coords),
        key=lambda x: preferred_order.index(x) if x in preferred_order else 999
    )

    for coord_name in sorted_coords:
        coord = da.coords[coord_name]
        if coord.size > 0:
            try:
                # Check for time coordinate
                is_datetime = np.issubdtype(coord.dtype, np.datetime64)
                is_time_name = 'time' in str(coord_name).lower()
                is_time = is_datetime or is_time_name

                if coord.size == 1:
                    # Single value coordinate
                    val = coord.values.item()
                    if is_datetime:
                        try:
                            import pandas as pd
                            # Convert numpy int (ns) back to timestamp
                            val = pd.to_datetime(val)
                        except Exception:
                            pass
                        val_str = format_time_value(val)
                    elif is_time_name:
                        val_str = format_time_value(val)
                    else:
                        val_str = str(val)
                    
                    # Add units if available
                    coord_units = coord.attrs.get('units', '')
                    units_str = f" ({coord_units})" if coord_units else ""
                    
                    lines.append(f"- **{coord_name}**: {val_str}{units_str}\n")
                else:
                    # Range of values
                    val_min = coord.min().values
                    val_max = coord.max().values

                    if is_time:
                        min_str = format_time_value(val_min)
                        max_str = format_time_value(val_max)
                        lines.append(f"- **{coord_name}**: [{min_str}, {max_str}]\n")
                    else:
                        # Numeric range
                        val_min_f = float(val_min)
                        val_max_f = float(val_max)
                        coord_units = coord.attrs.get('units', '')
                        units_str = f" ({coord_units})" if coord_units else ""
                        lines.append(f"- **{coord_name}**: [{val_min_f:.2f}, {val_max_f:.2f}]{units_str}\n")

            except Exception:
                lines.append(f"- **{coord_name}**: (complex coordinate)\n")

    return "".join(lines)


def format_data_size(size_bytes: int) -> str:
    """
    Format data size in human-readable format.

    Parameters
    ----------
    size_bytes : int
        Size in bytes

    Returns
    -------
    str
        Formatted size string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def summarize_dataset(ds: xr.Dataset) -> str:
    """
    Create a summary string for an xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to summarize

    Returns
    -------
    str
        Summary string
    """
    lines = [f"Dataset with {len(ds.data_vars)} variables:\n"]

    for var_name in ds.data_vars:
        var = ds[var_name]
        shape_str = " × ".join(str(s) for s in var.shape)
        lines.append(f"  - {var_name}: {shape_str}\n")

    return "".join(lines)
