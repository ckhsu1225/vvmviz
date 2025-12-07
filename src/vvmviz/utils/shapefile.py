"""
Shapefile Loading Utilities

Functions for loading and caching geographic boundary data from shapefiles.
"""

import logging
import shapefile
import holoviews as hv
import numpy as np
from pathlib import Path
from typing import Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=10)
def load_boundary_paths(
    shp_path: Path,
    cache_key: str,
    color: str = 'black',
    line_width: float = 1.0
) -> hv.Path:
    """
    Load boundary paths from a shapefile with caching.

    This function reads polygon boundaries from a shapefile and converts them
    to HoloViews Path objects for overlay on maps. Results are cached to avoid
    repeated file I/O.

    Parameters
    ----------
    shp_path : Path
        Path to the shapefile (.shp file)
    cache_key : str
        Unique key for caching (e.g., 'county_black_1.0')
    color : str, optional
        Line color (default: 'black')
    line_width : float, optional
        Line width (default: 1.0)

    Returns
    -------
    hv.Path
        HoloViews Path object containing boundary polygons

    Raises
    ------
    FileNotFoundError
        If shapefile does not exist
    RuntimeError
        If shapefile cannot be read

    Examples
    --------
    >>> from pathlib import Path
    >>> county_path = Path('/data/shapefiles/county.shp')
    >>> paths = load_boundary_paths(county_path, 'county', color='red', line_width=2.0)
    """
    if not shp_path.exists():
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")

    try:
        # Read shapefile
        sf = shapefile.Reader(str(shp_path))

        # Extract polygon segments
        path_segments = []

        for shape_rec in sf.shapeRecords():
            shape = shape_rec.shape

            # Handle different geometry types
            if shape.shapeType in [5, 15, 25]:  # Polygon types
                # Extract points from the shape
                points = shape.points
                parts = list(shape.parts) + [len(points)]

                # Split into separate polygons if there are multiple parts
                for i in range(len(parts) - 1):
                    start = parts[i]
                    end = parts[i + 1]
                    polygon_points = points[start:end]

                    if len(polygon_points) >= 3:  # Valid polygon needs at least 3 points
                        path_segments.append(polygon_points)

        if not path_segments:
            logger.warning(f"No valid polygons found in {shp_path}")
            # Return empty path
            return hv.Path([]).opts(color=color, line_width=line_width)

        # Merge all segments with NaN separators for efficient rendering
        total_points = sum(len(seg) for seg in path_segments)
        total_len = total_points + len(path_segments)  # +1 NaN per segment
        merged_data = np.full((total_len, 2), np.nan)

        current_idx = 0
        for seg in path_segments:
            seg_array = np.array(seg) if not isinstance(seg, np.ndarray) else seg
            n = len(seg_array)
            merged_data[current_idx:current_idx + n] = seg_array
            current_idx += n + 1  # Skip one for NaN separator

        # Create HoloViews Path object with merged data
        paths = hv.Path(
            [merged_data],
            kdims=['lon', 'lat']
        ).opts(
            color=color,
            line_width=line_width,
            tools=['hover']
        )

        return paths

    except Exception as e:
        raise RuntimeError(f"Failed to load shapefile {shp_path}: {e}")


def validate_shapefile(shp_path: Path) -> Tuple[bool, str]:
    """
    Validate that a shapefile exists and is readable.

    Parameters
    ----------
    shp_path : Path
        Path to shapefile

    Returns
    -------
    tuple[bool, str]
        (is_valid, message)
        - is_valid: True if shapefile is valid
        - message: Success message or error description
    """
    if not shp_path.exists():
        return False, f"Shapefile not found: {shp_path}"

    # Check for required companion files
    base_path = shp_path.with_suffix('')
    required_exts = ['.shp', '.shx', '.dbf']

    missing_files = []
    for ext in required_exts:
        if not (base_path.with_suffix(ext)).exists():
            missing_files.append(ext)

    if missing_files:
        return False, f"Missing required files: {', '.join(missing_files)}"

    # Try to read the shapefile
    try:
        sf = shapefile.Reader(str(shp_path))
        num_shapes = len(sf.shapes())
        return True, f"Valid shapefile with {num_shapes} shapes"
    except Exception as e:
        return False, f"Failed to read shapefile: {e}"
