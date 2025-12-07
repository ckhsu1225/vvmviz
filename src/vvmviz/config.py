"""
VVMViz Configuration Module

This module contains all global configuration, constants, and default settings
for the VVMViz application.
"""

import logging
import threading
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# Thread Safety
# =============================================================================

# Global lock for NetCDF I/O operations to prevent HDF5 segfaults
FILE_IO_LOCK = threading.RLock()


# =============================================================================
# Default Paths
# =============================================================================

# VVM data directory
DEFAULT_VVM_DIR = Path('/data2/VVM/taiwanvvm_summer/')

# Taiwan shapefile paths
TWCOUNTY_SHP_PATH = Path('/data/ckhsu/tw_county_town/county/COUNTY_MOI_1140318.shp')
TWTOWN_SHP_PATH = Path('/data/ckhsu/tw_county_town/town/TOWN_MOI_1140318.shp')


# =============================================================================
# Cache Settings
# =============================================================================

# Maximum number of frames to keep in memory cache
MAX_FRAME_CACHE_SIZE = 200

# Maximum number of datasets to cache (for @lru_cache decorator)
DATASET_CACHE_SIZE = 10


# =============================================================================
# Variable Names
# =============================================================================

TERRAIN_VAR_NAME = "terrain_height"


# =============================================================================
# Colormap Definitions (imported from plotting module)
# =============================================================================

# Import colormap definitions from plotting.colormaps (single source of truth)
# This allows the plotting module to be self-contained while config can still
# access these definitions if needed for backwards compatibility
from vvmviz.plotting.colormaps import (
    NCL_CMAP_CATEGORIES,
    ALL_NCL_CMAPS,
    DEFAULT_COLORMAP,
    VARIABLE_DEFAULTS,
    get_variable_default,
)


# =============================================================================
# Configuration Dataclass (Optional, for future extensibility)
# =============================================================================

@dataclass
class VVMVizConfig:
    """
    VVMViz configuration with sensible defaults.

    This can be extended in the future to support loading from
    external configuration files (e.g., config.toml).
    """

    # Default paths
    default_vvm_dir: Path = DEFAULT_VVM_DIR
    county_shapefile: Path = TWCOUNTY_SHP_PATH
    town_shapefile: Path = TWTOWN_SHP_PATH

    # Cache settings
    max_frame_cache_size: int = MAX_FRAME_CACHE_SIZE
    dataset_cache_size: int = DATASET_CACHE_SIZE

    # UI defaults
    default_colormap: str = DEFAULT_COLORMAP

    @classmethod
    def load_from_file(cls, config_path: Path | None = None) -> 'VVMVizConfig':
        """
        Load configuration from TOML file if exists, otherwise use defaults.

        Parameters
        ----------
        config_path : Path, optional
            Path to configuration file. If None, searches for config.toml
            in current directory or ~/.vvmviz/

        Returns
        -------
        VVMVizConfig
            Configuration instance
        """
        # Search paths: specified path -> ./config.toml -> ~/.vvmviz/config.toml
        search_paths = [
            config_path,
            Path.cwd() / 'config.toml',
            Path.home() / '.vvmviz' / 'config.toml'
        ]

        for path in search_paths:
            if path and path.exists():
                try:
                    import tomllib
                    with open(path, 'rb') as f:
                        data = tomllib.load(f)
                        vvmviz_config = data.get('vvmviz', {})

                        # Create instance with loaded values
                        return cls(
                            default_vvm_dir=vvmviz_config.get('default_vvm_dir', cls.default_vvm_dir),
                            county_shapefile=Path(vvmviz_config.get('county_shapefile', cls.county_shapefile)),
                            town_shapefile=Path(vvmviz_config.get('town_shapefile', cls.town_shapefile)),
                            max_frame_cache_size=vvmviz_config.get('max_frame_cache_size', cls.max_frame_cache_size),
                            dataset_cache_size=vvmviz_config.get('dataset_cache_size', cls.dataset_cache_size),
                            default_colormap=vvmviz_config.get('default_colormap', cls.default_colormap),
                        )
                except Exception as e:
                    logger.warning(f"Failed to load config from {path}: {e}")
                    # Fall through to use defaults

        # No config file found or loading failed, use defaults
        return cls()


# =============================================================================
# Global Config Instance
# =============================================================================

# Default configuration instance (can be overridden by loading from file)
config = VVMVizConfig()
