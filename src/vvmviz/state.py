"""
VVMViz Application State Module

This module provides the global application state management for VVMViz,
including the AppState class and range tracking utilities.

The state is centralized here to:
- Keep all application state in one place
- Enable easy sharing between Controller and UI components
- Avoid circular imports
"""

import logging

import param
import holoviews as hv

logger = logging.getLogger(__name__)


class AppState(param.Parameterized):
    """
    Global application state manager.

    This class holds the current application state including:
    - Current simulation path
    - Variable groups
    - Loaded data
    - View state (zoom ranges, etc.)

    Attributes
    ----------
    current_sim_path : str or None
        Path to the currently selected simulation
    variable_groups : dict
        Mapping of category names to variable lists
    last_plot_object : object
        Reference to the last created plot
    saved_x_range : tuple or None
        Saved x-axis range for zoom persistence
    saved_y_range : tuple or None
        Saved y-axis range for zoom persistence
    current_coord_bounds : tuple or None
        (lon_min, lon_max, lat_min, lat_max) of current simulation
    auto_contour_range : bool
        Whether contour range is in auto mode
    is_loading_simulation : bool
        Flag to prevent auto-plotting during simulation load
    has_data_loaded : bool
        Flag to track if data has been explicitly loaded
    skip_range_extraction : bool
        Flag to skip range extraction during reset
    lev_vals : array or None
        Cached vertical level values for index lookup
    """

    # Simulation info
    current_sim_path = param.String(default=None, allow_None=True)
    variable_groups = param.Dict(default={})
    
    # Plot reference
    last_plot_object = param.Parameter(default=None)

    # View state (for preserving zoom/pan)
    saved_x_range = param.Parameter(default=None, allow_None=True)
    saved_y_range = param.Parameter(default=None, allow_None=True)

    # Coordinate bounds for detecting domain changes
    current_coord_bounds = param.Parameter(default=None, allow_None=True)

    # Contour state
    auto_contour_range = param.Boolean(default=True)
    updating_contour_programmatically = param.Boolean(default=False)
    
    # Loading flags
    is_loading_simulation = param.Boolean(default=False)
    has_data_loaded = param.Boolean(default=False)
    skip_range_extraction = param.Boolean(default=False)
    
    # Vertical levels cache
    lev_vals = param.Parameter(default=None)
    
    # Time index mapping (slider position -> actual file index)
    time_index_map = param.Dict(default={})
    available_time_indices = param.List(default=[])


# =============================================================================
# Range Tracking
# =============================================================================

# Global RangeXY stream to persist zoom/pan across updates
range_stream = hv.streams.RangeXY()


def create_range_recorder(app_state: AppState):
    """
    Create a callback to persist user zoom/pan ranges to app_state.

    This callback is subscribed to the range_stream and updates app_state
    whenever the user zooms or pans on the plot.

    Parameters
    ----------
    app_state : AppState
        The application state instance to update

    Returns
    -------
    callable
        The callback function
    """
    def record_ranges(**ranges):
        xr = ranges.get('x_range')
        yr = ranges.get('y_range')

        if xr and all(v is not None for v in xr):
            if app_state.saved_x_range != xr:
                app_state.saved_x_range = xr
                logger.info(f"[ZOOM] Recorded x_range: {xr}")

        if yr and all(v is not None for v in yr):
            if app_state.saved_y_range != yr:
                app_state.saved_y_range = yr
                logger.info(f"[ZOOM] Recorded y_range: {yr}")

    return record_ranges


# Track the bokeh plot reference for range extraction
_bokeh_plot_ref = {'plot': None}


def get_bokeh_plot_ref():
    """Get the current Bokeh plot reference."""
    return _bokeh_plot_ref


def set_bokeh_plot_ref(plot):
    """Set the Bokeh plot reference."""
    _bokeh_plot_ref['plot'] = plot
