"""
Colormap Management Module

This module provides colormap-related functionality for VVMViz visualization:
- NCL (NCAR Command Language) colormap definitions and categories
- Variable-specific default colormap settings
- Colormap conversion utilities (matplotlib to hex palettes)
- Colormap resolution and reversal logic
"""

import logging
import fnmatch
from typing import Dict, List, Any

import matplotlib.pyplot as plt

try:
    import cmaps  # NCL colormaps
except ImportError:
    cmaps = None

logger = logging.getLogger(__name__)


# =============================================================================
# NCL Colormap Categories
# =============================================================================

NCL_CMAP_CATEGORIES = {
    "Rainbow": [
        "BlAqGrYeOrRe",
        "BlAqGrYeOrReVi200",
        "BlGrYeOrReVi200",
        "GMT_seis_r",
        "GMT_wysiwygcont",
        "MPL_jet",
        "MPL_rainbow",
        "NCV_bright",
        "MPL_hsv",
        "WhBlGrYeRe",
        "WhiteBlueGreenYellowRed"
    ],
    "Earth/Ocean": [
        "cmocean_deep",
        "cmp_haxby_r",
        "GMT_drywet",
        "OceanLakeLandSnow"
    ],
    "MeteoSwiss": [
        "precip2_17lev",
        "precip3_16lev"
    ],
    "Blue/Red": [
        "MPL_bwr",
        "MPL_coolwarm",
        "MPL_RdBu_r",
        "MPL_seismic",
        "BlueWhiteOrangeRed",
        "NCV_jaisnd"
    ],
    "Vegetation": [
        "MPL_BrBG",
        "NEO_div_vegetation_a"
    ],
    "Blue/Green": [
        "cmocean_haline",
        "MPL_BuGn",
        "MPL_GnBu",
        "MPL_viridis",
        "MPL_YlGnBu",
        "WhiteBlue",
        "MPL_Purples",
        "WhiteGreen"
    ],
    "Red/Orange": [
        "MPL_Oranges",
        "MPL_Reds",
        "WhiteYellowOrangeRed"
    ],
    "Gray": [
        "MPL_Greys"
    ]
}

# Flatten all colormaps into a single list
ALL_NCL_CMAPS = [cmap for category in NCL_CMAP_CATEGORIES.values() for cmap in category]

# Default colormap
DEFAULT_COLORMAP = 'MPL_GnBu' if 'MPL_GnBu' in ALL_NCL_CMAPS else ALL_NCL_CMAPS[0]


# =============================================================================
# Variable-Specific Default Settings
# =============================================================================

VARIABLE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # Outgoing longwave radiation
    'olr': {
        'cmap': 'MPL_GnBu',
        'reverse': False
    },

    # Surface precipitation
    'sprec': {
        'cmap': 'precip3_16lev'
    },

    # Wind components (u, v, w) - use diverging colormaps with symmetric range
    'u': {
        'cmap': 'MPL_RdBu_r',
        'symmetric': True
    },
    'v': {
        'cmap': 'MPL_RdBu_r',
        'symmetric': True
    },
    'w': {
        'cmap': 'MPL_RdBu_r',
        'symmetric': True
    },

    # Vorticity components - use diverging colormaps
    'zeta': {
        'cmap': 'MPL_seismic',
        'symmetric': True
    },
    'eta': {
        'cmap': 'MPL_seismic',
        'symmetric': True
    },
    'xi': {
        'cmap': 'MPL_seismic',
        'symmetric': True
    },

    # Hydrometeor variables
    'brim': {'cmap': 'precip3_16lev'},
    'qc': {'cmap': 'precip3_16lev'},
    'qi': {'cmap': 'precip3_16lev'},
    'qr': {'cmap': 'precip3_16lev'},
    'qrim': {'cmap': 'precip3_16lev'},
    'nc': {'cmap': 'precip3_16lev'},
    'ni': {'cmap': 'precip3_16lev'},
    'nr': {'cmap': 'precip3_16lev'},

    # Water vapor
    'qv': {'cmap': 'MPL_GnBu'},

    # Temperature variables
    'th': {'cmap': 'MPL_jet'},
    't': {'cmap': 'MPL_jet'},
    'tv': {'cmap': 'MPL_jet'},
    'the': {'cmap': 'MPL_jet'},
    'thes': {'cmap': 'MPL_jet'},
    'thv': {'cmap': 'MPL_jet'},

    # Tracer variables (wildcard pattern)
    'tr*': {'cmap': 'WhiteBlueGreenYellowRed'},

    # Column-integrated variables
    'cwv': {'cmap': 'GMT_drywet'},
    'iwp': {'cmap': 'precip3_16lev'},
    'lwp': {'cmap': 'precip3_16lev'},

    # Other variables
    'hm': {'cmap': 'MPL_jet'},
    'hms': {'cmap': 'MPL_jet'},
    'qvs': {'cmap': 'MPL_jet'},
    'rh': {'cmap': 'MPL_BrBG'},
    'sd': {'cmap': 'MPL_jet'},
    'ws': {'cmap': 'WhiteBlueGreenYellowRed'},

    # Terrain
    'terrain_height': {'cmap': 'OceanLakeLandSnow'}
}


# =============================================================================
# Colormap Lookup Functions
# =============================================================================

def get_variable_default(var_name: str) -> Dict[str, Any]:
    """
    Get default colormap settings for a specific variable.

    Supports wildcard pattern matching (e.g., 'tr*' matches 'tr1', 'tr2', etc.)

    Parameters
    ----------
    var_name : str
        Variable name to look up

    Returns
    -------
    Dict[str, Any]
        Dictionary containing default settings:
        - 'cmap': Colormap name
        - 'reverse': Whether to reverse the colormap (optional)
        - 'symmetric': Whether to use symmetric color limits (optional)

    Examples
    --------
    >>> defaults = get_variable_default('u')
    >>> defaults['cmap']
    'MPL_RdBu_r'
    >>> defaults['symmetric']
    True

    >>> defaults = get_variable_default('tr1')  # Matches 'tr*' pattern
    >>> defaults['cmap']
    'WhiteBlueGreenYellowRed'
    """
    # Direct match
    if var_name in VARIABLE_DEFAULTS:
        return VARIABLE_DEFAULTS[var_name].copy()

    # Wildcard match
    for pattern, defaults in VARIABLE_DEFAULTS.items():
        if '*' in pattern and fnmatch.fnmatch(var_name, pattern):
            return defaults.copy()

    return {}


def resolve_colormap(
    cmap_name: str,
    reverse: bool = False
) -> Any:
    """
    Resolve a colormap name to an actual colormap object.

    This function handles:
    1. NCL colormap lookup (via cmaps module)
    2. Matplotlib colormap lookup
    3. Colormap reversal

    Parameters
    ----------
    cmap_name : str
        Name of the colormap to resolve
    reverse : bool, default=False
        Whether to reverse the colormap

    Returns
    -------
    colormap object or str
        - NCL colormap object (if found in cmaps)
        - Matplotlib colormap object (if found in matplotlib)
        - Modified name string (if reversal requested but object reversal fails)

    Examples
    --------
    >>> cmap = resolve_colormap('viridis')
    >>> type(cmap)
    <class 'matplotlib.colors.LinearSegmentedColormap'>

    >>> cmap_reversed = resolve_colormap('viridis', reverse=True)
    >>> cmap_reversed.name
    'viridis_r'
    """
    cmap = cmap_name

    # Step 1: Resolve NCL colormap if it's a string and cmaps is available
    if isinstance(cmap, str) and cmaps is not None and hasattr(cmaps, cmap):
        try:
            cmap = getattr(cmaps, cmap)
            logger.debug(f"Resolved NCL colormap: {cmap_name}")
        except Exception as e:
            logger.warning(f"Failed to load NCL colormap {cmap_name}: {e}")

    # Step 2: Handle reversal
    if reverse:
        try:
            # Try object reversal (works for matplotlib and NCL colormaps)
            if isinstance(cmap, str):
                # If still a string, use matplotlib
                cmap_obj = plt.get_cmap(cmap)
                cmap = cmap_obj.reversed()
                logger.debug(f"Reversed matplotlib colormap: {cmap_name}")
            else:
                # Already an object (NCL or matplotlib)
                cmap = cmap.reversed()
                logger.debug(f"Reversed colormap object: {cmap_name}")
        except Exception as e:
            # Fallback: modify string name
            logger.warning(f"Failed to reverse colormap {cmap_name}, using name modification: {e}")
            if isinstance(cmap, str):
                if cmap.endswith('_r'):
                    cmap = cmap[:-2]  # Remove '_r' suffix
                else:
                    cmap = cmap + '_r'  # Add '_r' suffix

    return cmap


def get_colormap_categories() -> Dict[str, List[str]]:
    """
    Get all available colormap categories.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping category names to lists of colormap names

    Examples
    --------
    >>> categories = get_colormap_categories()
    >>> 'Rainbow' in categories
    True
    >>> 'MPL_jet' in categories['Rainbow']
    True
    """
    return NCL_CMAP_CATEGORIES.copy()


def get_all_colormaps() -> List[str]:
    """
    Get a flat list of all available colormap names.

    Returns
    -------
    List[str]
        List of all colormap names across all categories

    Examples
    --------
    >>> all_cmaps = get_all_colormaps()
    >>> 'WhiteBlueGreenYellowRed' in all_cmaps
    True
    """
    return ALL_NCL_CMAPS.copy()
