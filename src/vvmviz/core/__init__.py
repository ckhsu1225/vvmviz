"""
VVMViz Core Module

Core data loading and processing functionality for VVM simulations.
"""

from vvmviz.core.data_loader import (
    list_simulations,
    scan_variable_groups,
    enrich_variable_groups,
    open_dataset,
    get_terrain_data,
    get_coordinate_info,
    get_vertical_info,
    get_terrain_info,
    scan_time_indices,
)

from vvmviz.core.data_processor import (
    get_data_array,
    get_wind_vectors,
    get_contour_data,
    load_frame_bundle,
    squeeze_singleton_dims,
    select_single_time_level,
)

__all__ = [
    # Data Loading
    'list_simulations',
    'scan_variable_groups',
    'enrich_variable_groups',
    'open_dataset',
    'get_terrain_data',
    'get_coordinate_info',
    'get_vertical_info',
    'get_terrain_info',
    'scan_time_indices',
    # Data Processing
    'get_data_array',
    'get_wind_vectors',
    'get_contour_data',
    'load_frame_bundle',
    'squeeze_singleton_dims',
    'select_single_time_level',
]
