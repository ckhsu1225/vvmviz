"""
VVMViz Utilities

Utility functions and classes for caching, metadata formatting,
and shapefile handling.
"""

from vvmviz.utils.cache import CacheManager, FrameRequest, get_cache_manager
from vvmviz.utils.metadata import (
    format_time_value,
    extract_metadata_from_dataarray,
    build_metadata_markdown,
    format_data_size,
    summarize_dataset
)
from vvmviz.utils.shapefile import (
    load_boundary_paths,
    validate_shapefile
)

__all__ = [
    # Cache
    'CacheManager',
    'FrameRequest',
    'get_cache_manager',
    # Metadata
    'format_time_value',
    'extract_metadata_from_dataarray',
    'build_metadata_markdown',
    'format_data_size',
    'summarize_dataset',
    # Shapefile
    'load_boundary_paths',
    'validate_shapefile',
]
