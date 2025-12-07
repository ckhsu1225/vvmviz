"""
VVMViz Plotting Package

This package provides all plotting and visualization functionality:
- Colormap management (colormaps.py)
- Overlay rendering (overlays.py)
- Core plotting functions (base.py)
"""

# Colormap functions
from vvmviz.plotting.colormaps import (
    # Colormap categories and lists
    NCL_CMAP_CATEGORIES,
    ALL_NCL_CMAPS,
    DEFAULT_COLORMAP,
    VARIABLE_DEFAULTS,

    # Utility functions
    get_variable_default,
    resolve_colormap,
    get_colormap_categories,
    get_all_colormaps,
)

# Overlay functions
from vvmviz.plotting.overlays import (
    create_wind_vectors,
    create_contour_overlay,
    get_county_boundaries,
    get_town_boundaries,
)

# Core plotting functions
from vvmviz.plotting.base import (
    calculate_color_limits,
    create_image,
    compose_plot,
    apply_ranges,
    create_main_plot,
)

__all__ = [
    # Colormap exports
    'NCL_CMAP_CATEGORIES',
    'ALL_NCL_CMAPS',
    'DEFAULT_COLORMAP',
    'VARIABLE_DEFAULTS',
    'get_variable_default',
    'resolve_colormap',
    'get_colormap_categories',
    'get_all_colormaps',

    # Overlay exports
    'create_wind_vectors',
    'create_contour_overlay',
    'get_county_boundaries',
    'get_town_boundaries',

    # Core plotting exports
    'calculate_color_limits',
    'create_image',
    'compose_plot',
    'apply_ranges',
    'create_main_plot',
]
