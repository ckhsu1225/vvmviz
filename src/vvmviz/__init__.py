"""
VVMViz - Interactive VVM Data Visualization

A modular, interactive dashboard for visualizing Vector Vorticity Model (VVM)
simulation data.

This package provides:
- Data loading and processing for VVM NetCDF files
- Interactive visualization with Panel and HoloViews
- Configurable plotting with NCL colormaps
- Caching and prefetching for performance
- Overlay support (wind vectors, contours, boundaries)

Quick Start
-----------
Create a dashboard:

>>> import vvmviz
>>> from vvmviz.ui import create_dashboard
>>> from vvmviz.core.data_loader import scan_variable_groups
>>>
>>> # Scan variables
>>> groups = scan_variable_groups('/data2/VVM/sim001/')
>>>
>>> # Create and serve dashboard
>>> dashboard = create_dashboard(groups)
>>> dashboard.servable()

Or use the provided app.py:

    $ panel serve app.py --show --port 5006
"""

__version__ = "0.1.0"

# Core components
from vvmviz.config import config, VVMVizConfig

# Data layer
from vvmviz.core.data_loader import (
    list_simulations,
    scan_variable_groups,
    open_dataset,
    get_terrain_data
)
from vvmviz.core.data_processor import (
    get_data_array,
    get_wind_vectors,
    load_frame_bundle
)

# Plotting layer
from vvmviz.plotting.base import create_main_plot
from vvmviz.plotting.colormaps import (
    get_variable_default,
    resolve_colormap,
    NCL_CMAP_CATEGORIES
)
from vvmviz.plotting.overlays import (
    create_wind_vectors,
    create_contour_overlay,
    get_county_boundaries,
    get_town_boundaries
)

# Utilities
from vvmviz.utils.cache import get_cache_manager, CacheManager
from vvmviz.utils.metadata import build_metadata_markdown, format_time_value

# UI components
from vvmviz.ui import create_dashboard, DomainMapSelector, PlaybackController

__all__ = [
    # Version
    '__version__',

    # Config
    'config',
    'VVMVizConfig',

    # Data loading
    'list_simulations',
    'scan_variable_groups',
    'open_dataset',
    'get_terrain_data',

    # Data processing
    'get_data_array',
    'get_wind_vectors',
    'load_frame_bundle',

    # Plotting
    'create_main_plot',
    'resolve_colormap',
    'get_variable_default',
    'NCL_CMAP_CATEGORIES',

    # Overlays
    'create_wind_vectors',
    'create_contour_overlay',
    'get_county_boundaries',
    'get_town_boundaries',

    # Utilities
    'get_cache_manager',
    'CacheManager',
    'build_metadata_markdown',
    'format_time_value',

    # UI
    'create_dashboard',
    'DomainMapSelector',
    'PlaybackController'
]
