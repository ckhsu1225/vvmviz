"""
VVMViz Application Controller

This module provides the main Controller class that orchestrates all
application logic for the VVMViz dashboard.

The VVMVizController follows the Controller pattern, separating:
- View (UI widgets, layout) - in vvmviz.ui
- Model (data loading, processing) - in vvmviz.core
- Controller (this module) - orchestrates View and Model

This design improves:
- Readability: All application logic is in one place
- Maintainability: Easy to trace state changes and debug
- Extensibility: New features = new methods, minimal changes elsewhere
"""

import os
import logging
from typing import Dict, Any, Optional

import numpy as np
import panel as pn
import holoviews as hv

from vvmviz.config import FILE_IO_LOCK
from vvmviz.state import AppState, range_stream, create_range_recorder
from vvmviz.core.data_loader import (
    list_simulations,
    scan_variable_groups,
    scan_time_indices,
    get_vertical_info,
    get_coordinate_info
)
from vvmviz.core.data_processor import (
    load_frame_bundle,
    squeeze_singleton_dims,
    get_data_array
)
from vvmviz.plotting.base import create_main_plot, calculate_color_limits
from vvmviz.plotting.overlays import (
    create_wind_vectors,
    create_contour_overlay,
    get_county_boundaries,
    get_town_boundaries
)
from vvmviz.plotting.colormaps import get_variable_default, DEFAULT_COLORMAP
from vvmviz.utils.cache import get_cache_manager, FrameRequest
from vvmviz.utils.metadata import build_metadata_markdown, format_time_value
from vvmviz.ui import DomainMapSelector

logger = logging.getLogger(__name__)


class VVMVizController:
    """
    Main application controller for VVMViz dashboard.

    This class orchestrates all user interactions, data loading,
    and plot updates. It connects the UI (widgets) with the data
    processing layer.

    Parameters
    ----------
    widgets : dict
        Dictionary of all widgets created by create_all_widgets()
    plot_pane : pn.pane.HoloViews
        The pane for displaying the main plot
    metadata_pane : pn.pane.Markdown
        The pane for displaying metadata
    map_selector : DomainMapSelector
        The interactive domain map selector

    Attributes
    ----------
    state : AppState
        The global application state
    cache : CacheManager
        The frame cache manager
    widgets : dict
        Reference to all UI widgets
    plot_pane : pn.pane.HoloViews
        Reference to the plot display pane
    metadata_pane : pn.pane.Markdown
        Reference to the metadata pane
    map_selector : DomainMapSelector
        Reference to the domain map selector

    Example
    -------
    >>> from vvmviz.ui import create_dashboard
    >>> from vvmviz.controllers import VVMVizController
    >>>
    >>> layout = create_dashboard({})
    >>> controller = VVMVizController(
    ...     widgets=layout._vvmviz_widgets,
    ...     plot_pane=layout._vvmviz_plot_pane,
    ...     metadata_pane=layout._vvmviz_metadata_pane,
    ...     map_selector=layout._vvmviz_map_selector
    ... )
    >>> controller.attach_callbacks()
    """

    def __init__(
        self,
        widgets: Dict[str, Any],
        plot_pane: pn.pane.HoloViews,
        metadata_pane: pn.pane.Markdown,
        map_selector: DomainMapSelector
    ):
        # Core references
        self.widgets = widgets
        self.plot_pane = plot_pane
        self.metadata_pane = metadata_pane
        self.map_selector = map_selector

        # State and cache
        self.state = AppState()
        self.cache = get_cache_manager()

        # Setup range tracking
        self._range_recorder = create_range_recorder(self.state)
        range_stream.add_subscriber(self._range_recorder)

        logger.info("VVMVizController initialized")

    # =========================================================================
    # Simulation Management
    # =========================================================================

    def load_simulations(self, event=None):
        """
        Load simulations from the path specified in path_input widget.

        This is triggered by clicking the "Load" button next to the path input.
        It scans the directory for valid VVM simulations and populates the
        simulation selector.
        """
        path = self.widgets['path_input'].value

        if not os.path.isdir(path):
            if pn.state.notifications:
                pn.state.notifications.error(f"Invalid directory: {path}")
            logger.error(f"Invalid directory: {path}")
            return

        try:
            sims = list_simulations(path)

            if not sims:
                if pn.state.notifications:
                    pn.state.notifications.warning(f"No simulations found in {path}")
                logger.warning(f"No simulations found in {path}")
                self.widgets['sim_selector'].options = {}
                self.widgets['sim_selector'].value = None
                return

            # Update simulation selector
            new_options = {p.name: str(p) for p in sims}
            self.widgets['sim_selector'].options = new_options
            self.widgets['sim_selector'].value = list(new_options.values())[0]

            if pn.state.notifications:
                pn.state.notifications.success(f"Loaded {len(sims)} simulations.")
            logger.info(f"Loaded {len(sims)} simulations from {path}")

        except Exception as e:
            if pn.state.notifications:
                pn.state.notifications.error(f"Error loading simulations: {e}")
            logger.error(f"Error loading simulations: {e}", exc_info=True)

    def on_simulation_change(self, event):
        """
        Handle simulation selection change.

        Updates variable groups, range sliders, and domain map when
        a new simulation is selected.

        Parameters
        ----------
        event : param.Event
            The event object containing the new simulation path
        """
        sim_path = event.new

        if not sim_path:
            return

        try:
            # Set loading flag to prevent auto-plotting
            self.state.is_loading_simulation = True
            logger.info(f"Loading simulation: {sim_path}")

            # Scan variable groups
            groups = scan_variable_groups(sim_path)
            self.state.variable_groups = groups
            self.state.current_sim_path = sim_path

            # Update variable selectors
            self._update_variable_selectors(groups)

            # Update range sliders
            self._update_range_sliders(sim_path)

            # Update domain map
            self.map_selector.create_terrain_map(sim_path)

            logger.info(f"Simulation loaded: {len(groups)} variable groups found")

        except Exception as e:
            if pn.state.notifications:
                pn.state.notifications.error(f"Error loading simulation: {e}")
            logger.error(f"Error loading simulation: {e}", exc_info=True)
        finally:
            self.state.is_loading_simulation = False

    def _update_variable_selectors(self, groups: Dict[str, list]):
        """Update category and variable selectors with new groups."""
        category_options = list(groups.keys())
        self.widgets['var_selectors']['category'].options = category_options

        # Try to retain current selection if valid
        current_cat = self.widgets['var_selectors']['category'].value
        current_var = self.widgets['var_selectors']['variable'].value

        if current_cat in category_options:
            self.widgets['var_selectors']['category'].value = current_cat
            self.widgets['var_selectors']['variable'].options = groups[current_cat]

            if current_var in groups[current_cat]:
                self.widgets['var_selectors']['variable'].value = current_var
            else:
                self.widgets['var_selectors']['variable'].value = groups[current_cat][0]
        else:
            if category_options:
                self.widgets['var_selectors']['category'].value = category_options[0]

        # Update contour category if exists
        if 'contour_category' in self.widgets['var_selectors']:
            self.widgets['var_selectors']['contour_category'].options = category_options
            if self.widgets['var_selectors']['contour_category'].value not in category_options:
                if category_options:
                    self.widgets['var_selectors']['contour_category'].value = category_options[0]

    def _update_range_sliders(self, sim_path: str):
        """Update time, height, and spatial range sliders for new simulation."""
        # 1. Time
        available_t_indices = scan_time_indices(sim_path)
        self.state.available_time_indices = available_t_indices

        t_min = available_t_indices[0] if available_t_indices else 0
        t_max = available_t_indices[-1] if available_t_indices else 0
        self.widgets['range']['time'].start = t_min
        self.widgets['range']['time'].end = t_max
        self.widgets['range']['time'].value = (t_min, t_max)

        # 2. Vertical (Height)
        z_info = get_vertical_info(sim_path)
        if z_info and 'height_range' in z_info:
            z_min, z_max = z_info['height_range']
            if z_max <= z_min:
                z_max = z_min + 1000
            self.widgets['range']['lev'].start = float(z_min)
            self.widgets['range']['lev'].end = float(z_max)
            self.widgets['range']['lev'].value = (float(z_min), float(z_max))

            # Reset lev_slider to first available height
            if z_info.get('levels') and len(z_info['levels']) > 0:
                first_level = float(z_info['levels'][0])
                self.widgets['lev_slider'].options = {f"{first_level:.0f} m": first_level}
                self.widgets['lev_slider'].value = first_level
                logger.debug(f"Reset lev_slider to {first_level}m for new simulation")

        # 3. Horizontal (X, Y)
        coord_info = get_coordinate_info(sim_path)
        if coord_info:
            nx = coord_info.get('nx', 100)
            ny = coord_info.get('ny', 100)

            self.widgets['range']['x'].start = 0
            self.widgets['range']['x'].end = nx
            self.widgets['range']['x'].value = (0, nx)

            self.widgets['range']['y'].start = 0
            self.widgets['range']['y'].end = ny
            self.widgets['range']['y'].value = (0, ny)

            # Check grid dimension changes for zoom reset
            new_grid_bounds = (nx, ny)
            old_grid_bounds = getattr(self.state, '_temp_grid_bounds', None)

            if old_grid_bounds is not None and new_grid_bounds != old_grid_bounds:
                logger.info(f"Grid dimensions changed, resetting zoom")
                self.state.saved_x_range = None
                self.state.saved_y_range = None

            self.state._temp_grid_bounds = new_grid_bounds

    # =========================================================================
    # Variable Selection
    # =========================================================================

    def on_category_change(self, event):
        """
        Handle variable category change.

        Updates the variable list when user selects a different category.
        """
        category = event.new
        groups = self.state.variable_groups

        if category in groups:
            new_options = groups[category]
            self.widgets['var_selectors']['variable'].options = new_options

            # Keep current variable if exists in new category
            current_val = self.widgets['var_selectors']['variable'].value
            if current_val in new_options:
                self.widgets['var_selectors']['variable'].value = current_val
            else:
                self.widgets['var_selectors']['variable'].value = new_options[0]

    def on_contour_category_change(self, event):
        """Handle contour variable category change."""
        category = event.new
        groups = self.state.variable_groups

        if 'contour_variable' in self.widgets['var_selectors'] and category in groups:
            placeholder = "Select Variable..."
            new_options = [placeholder] + groups[category]
            self.widgets['var_selectors']['contour_variable'].options = new_options
            self.widgets['var_selectors']['contour_variable'].value = placeholder

    # =========================================================================
    # Data Loading
    # =========================================================================

    def load_data(self, event):
        """
        Load data for the selected variable.

        This is triggered by clicking the "Load Data" button. It:
        - Loads a sample to get metadata
        - Sets up time and level sliders
        - Applies variable-specific defaults (colormap, etc.)
        - Triggers the first plot update
        """
        var_name = self.widgets['var_selectors']['variable'].value
        sim_path = self.state.current_sim_path

        if not var_name or not sim_path:
            if pn.state.notifications:
                pn.state.notifications.warning("Please select a simulation and variable")
            return

        # Initialize defaults
        target_cmap = DEFAULT_COLORMAP
        target_reverse = False
        target_symmetric = False

        try:
            self.state.is_loading_simulation = True
            logger.info(f"Loading data: {var_name} from {sim_path}")

            # Ensure variable selector maintains correct value
            self.widgets['var_selectors']['variable'].value = var_name

            # Clear cache
            self.cache.clear()

            # Get ranges from widgets
            t_range_full = self.widgets['range']['time'].value
            z_range_full = self.widgets['range']['lev'].value
            x_range = self.widgets['range']['x'].value
            y_range = self.widgets['range']['y'].value

            # Load sample to get metadata
            sample_t = self._get_first_available_time(t_range_full)
            sample_t_range = (sample_t, sample_t)

            with FILE_IO_LOCK:
                da_sample = get_data_array(
                    sim_path, var_name,
                    t_range=sample_t_range,
                    z_range=z_range_full,
                    x_range=x_range,
                    y_range=y_range,
                    use_cache=False
                )

            if da_sample is None:
                if pn.state.notifications:
                    pn.state.notifications.error(f"Failed to load variable: {var_name}")
                return

            # Check coordinate bounds changes
            self._check_coordinate_bounds(da_sample)

            # Update metadata
            self.metadata_pane.object = build_metadata_markdown(da_sample)

            # Setup sliders
            self._setup_time_slider(da_sample)
            self._setup_level_slider(da_sample)

            # Apply variable defaults
            defaults = get_variable_default(var_name) or {}
            target_cmap = defaults.get('cmap', DEFAULT_COLORMAP)
            target_reverse = defaults.get('reverse', False)
            target_symmetric = defaults.get('symmetric', False)

            self.widgets['cmap_selector'].value = target_cmap
            self.widgets['cmap_reverse'].value = target_reverse
            self.widgets['clim']['symmetric'].value = target_symmetric
            self.widgets['clim']['lock'].value = False

            logger.info(f"Data loaded successfully: {var_name}")

            # Mark as loaded
            self.state.has_data_loaded = True

        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            if pn.state.notifications:
                pn.state.notifications.error(f"Data loading error: {e}")
        finally:
            # Trigger plot update
            self.update_plot(force=True)
            self.state.skip_range_extraction = False
            self.state.is_loading_simulation = False

    def _get_first_available_time(self, t_range):
        """Get first available time index within range."""
        all_available = self.state.available_time_indices
        t_start, t_end = t_range

        filtered = [i for i in all_available if t_start <= i <= t_end]
        if filtered:
            return filtered[0]
        elif all_available:
            return all_available[0]
        return t_start

    def _check_coordinate_bounds(self, da_sample):
        """Check and update coordinate bounds, reset ranges if changed."""
        try:
            new_bounds = {}

            if 'lon' in da_sample.coords and 'lat' in da_sample.coords:
                new_bounds['x'] = (float(da_sample.coords['lon'].min()),
                                   float(da_sample.coords['lon'].max()))
                new_bounds['y'] = (float(da_sample.coords['lat'].min()),
                                   float(da_sample.coords['lat'].max()))

            if 'lev' in da_sample.coords:
                new_bounds['z'] = (float(da_sample.coords['lev'].min()),
                                   float(da_sample.coords['lev'].max()))

            old_bounds = getattr(self.state, 'current_all_bounds', {})

            x_changed = old_bounds.get('x') != new_bounds.get('x') and old_bounds.get('x') is not None
            y_changed = old_bounds.get('y') != new_bounds.get('y') and old_bounds.get('y') is not None
            z_changed = old_bounds.get('z') != new_bounds.get('z') and old_bounds.get('z') is not None

            if x_changed or y_changed:
                logger.info(f"XY bounds changed, resetting zoom")
                self.state.saved_x_range = None
                self.state.saved_y_range = None
                self.state.skip_range_extraction = True

            if z_changed:
                logger.info(f"Z bounds changed, will reset level slider")
                self.state._reset_lev_slider = True

            self.state.current_all_bounds = new_bounds

            if 'x' in new_bounds and 'y' in new_bounds:
                self.state.current_coord_bounds = (
                    new_bounds['x'][0], new_bounds['x'][1],
                    new_bounds['y'][0], new_bounds['y'][1]
                )

        except Exception as e:
            logger.warning(f"Failed to extract coordinate bounds: {e}")

    def _setup_time_slider(self, da_sample):
        """Setup time slider based on loaded data."""
        if 'time' not in da_sample.dims:
            self._hide_time_controls()
            return

        all_available = self.state.available_time_indices
        if not all_available:
            t_range = self.widgets['range']['time'].value
            all_available = list(range(int(t_range[1] - t_range[0] + 1)))

        # Filter by time range
        t_range = self.widgets['range']['time'].value
        filtered_indices = [i for i in all_available if t_range[0] <= i <= t_range[1]]

        if not filtered_indices:
            logger.warning(f"No time indices in range {t_range}, using all")
            filtered_indices = all_available

        # Build options
        t_options = {}
        self.state.time_index_map = {}

        for slider_pos, actual_idx in enumerate(filtered_indices):
            label = f"Step {actual_idx}"
            t_options[label] = slider_pos
            self.state.time_index_map[slider_pos] = actual_idx

        self.widgets['time_controls']['slider'].options = t_options

        # Retain position if valid
        old_pos = self.widgets['time_controls']['slider'].value
        if old_pos is not None and old_pos < len(filtered_indices):
            new_pos = old_pos
        else:
            new_pos = 0

        self.widgets['time_controls']['slider'].value = new_pos
        self._show_time_controls()

    def _setup_level_slider(self, da_sample):
        """Setup level slider based on loaded data."""
        if 'lev' not in da_sample.dims:
            self.state.lev_vals = None
            self.state._reset_lev_slider = False
            self.widgets['lev_slider'].visible = False
            return

        lev_vals = da_sample.coords['lev'].values
        self.state.lev_vals = lev_vals

        lev_options = {f"{z:.0f} m": float(z) for z in lev_vals}
        self.widgets['lev_slider'].options = lev_options

        # Check if reset needed
        should_reset = getattr(self.state, '_reset_lev_slider', False)
        old_val = self.widgets['lev_slider'].value

        if should_reset or old_val not in lev_options.values():
            new_val = float(lev_vals[0])
            if should_reset:
                logger.info(f"Resetting level slider: {old_val} -> {new_val}")
        else:
            new_val = old_val

        self.widgets['lev_slider'].value = new_val
        self.state._reset_lev_slider = False
        self.widgets['lev_slider'].visible = True

    def _show_time_controls(self):
        """Show all time control widgets."""
        self.widgets['time_controls']['slider'].visible = True
        self.widgets['time_controls']['prev'].visible = True
        self.widgets['time_controls']['next'].visible = True
        self.widgets['time_controls']['play'].visible = True
        self.widgets['time_controls']['speed'].visible = True

    def _hide_time_controls(self):
        """Hide all time control widgets."""
        self.widgets['time_controls']['slider'].visible = False
        self.widgets['time_controls']['prev'].visible = False
        self.widgets['time_controls']['next'].visible = False
        self.widgets['time_controls']['play'].visible = False
        self.widgets['time_controls']['speed'].visible = False

    # =========================================================================
    # Plot Update
    # =========================================================================

    def update_plot(self, event=None, force=False):
        """
        Main plot update function.

        This is triggered by changes in time/level sliders, colormap settings,
        overlay toggles, etc. It loads the required data and creates the plot.

        Parameters
        ----------
        event : param.Event, optional
            The triggering event (if from a widget)
        force : bool, optional
            If True, force update even during loading
        """
        if not self.state.current_sim_path:
            return
        if self.state.is_loading_simulation and not force:
            return
        if not self.state.has_data_loaded:
            return

        try:
            pn.state.param.busy = True

            # Extract ranges from Bokeh if not skipping
            self._extract_bokeh_ranges()

            # Gather parameters from widgets
            params = self._gather_plot_params()
            if params is None:
                return

            # Load data bundle
            bundle = self._load_data_bundle(params, force)
            if bundle is None or 'main' not in bundle:
                return

            # Process main variable
            main_da = squeeze_singleton_dims(bundle['main'])

            # Prefetch next frame
            self._prefetch_next_frame(params)

            # Update clim if not locked
            self._update_clim_widgets(main_da)

            # Build dataset with hover info
            main_ds = main_da.to_dataset()
            main_ds.attrs['main_var'] = params['var_name']
            hover_dims = []

            # Prepare overlays
            overlays, c_title, hover_dims = self._prepare_overlays(bundle, params, main_ds, hover_dims)

            # Create title
            plot_title = self._build_plot_title(main_ds, params, c_title)

            # Create base plot
            base_plot = self._create_base_plot(main_ds, params, plot_title, hover_dims)

            # Attach range stream
            try:
                range_stream.source = base_plot
            except Exception as e:
                logger.warning(f"Failed to attach range_stream: {e}")

            # Build final plot with overlays
            final_plot = self._compose_final_plot(base_plot, overlays, bundle, params)

            # Update state and pane
            self.state.last_plot_object = final_plot
            self.plot_pane.object = final_plot

            # Trigger initial range event
            self._trigger_range_event()

            # Update metadata
            contour_da = squeeze_singleton_dims(bundle['contour']) if 'contour' in bundle else None
            self.metadata_pane.object = build_metadata_markdown(main_da, contour_da=contour_da)

        except Exception as e:
            logger.error(f"Error updating plot: {e}", exc_info=True)
            if pn.state.notifications:
                pn.state.notifications.error(f"Plot error: {e}")
        finally:
            pn.state.param.busy = False

    def _extract_bokeh_ranges(self):
        """Extract current ranges from Bokeh plot model."""
        if self.state.skip_range_extraction:
            logger.info("[ZOOM] Skipping range extraction (reset in progress)")
            return

        try:
            if hasattr(self.plot_pane, '_models') and self.plot_pane._models:
                bokeh_model = list(self.plot_pane._models.values())[0][0]
                if hasattr(bokeh_model, 'x_range') and hasattr(bokeh_model, 'y_range'):
                    current_x = (bokeh_model.x_range.start, bokeh_model.x_range.end)
                    current_y = (bokeh_model.y_range.start, bokeh_model.y_range.end)

                    if all(v is not None for v in current_x):
                        self.state.saved_x_range = current_x
                    if all(v is not None for v in current_y):
                        self.state.saved_y_range = current_y
        except Exception as e:
            logger.debug(f"Could not extract ranges from Bokeh: {e}")

    def _gather_plot_params(self) -> Optional[Dict[str, Any]]:
        """Gather all plot parameters from widgets."""
        var_name = self.widgets['var_selectors']['variable'].value
        if not var_name:
            return None

        # Time
        slider_pos = self.widgets['time_controls']['slider'].value
        t_val = self.state.time_index_map.get(slider_pos, slider_pos)

        # Level
        z_val = self.widgets['lev_slider'].value

        # Determine z_range
        if var_name in ['cwv', 'iwp', 'lwp']:
            z_info = get_vertical_info(self.state.current_sim_path)
            if z_info and 'height_range' in z_info:
                z_range = tuple(z_info['height_range'])
            else:
                z_range = tuple(self.widgets['range']['lev'].value)
        elif self.state.lev_vals is not None and len(self.state.lev_vals) > 0 and self.widgets['lev_slider'].visible:
            idx = int((np.abs(self.state.lev_vals - z_val)).argmin())
            z_range = ('index', idx, idx)
        else:
            z_range = None

        # Contour variable
        contour_var = self.widgets['var_selectors']['contour_variable'].value
        if contour_var == "Select Variable...":
            contour_var = None

        return {
            'sim_path': self.state.current_sim_path,
            'var_name': var_name,
            't_val': t_val,
            'z_val': z_val,
            'z_range': z_range,
            'x_range': self.widgets['range']['x'].value,
            'y_range': self.widgets['range']['y'].value,
            'wind_enabled': self.widgets['overlays']['wind'].value,
            'contour_enabled': self.widgets['overlays']['contour'].value,
            'contour_var': contour_var,
            'use_surface_wind': not self.widgets['lev_slider'].visible,
            'cmap_name': self.widgets['cmap_selector'].value,
            'reverse_cmap': self.widgets['cmap_reverse'].value,
            'symmetric_clim': self.widgets['clim']['symmetric'].value,
        }

    def _load_data_bundle(self, params: Dict[str, Any], force: bool = False):
        """Load data bundle from cache or file."""
        cache_key = (
            params['var_name'],
            (params['t_val'], params['t_val']),
            params['z_range'],
            params['wind_enabled'],
            params['contour_enabled'],
            params['contour_var']
        )

        bundle = self.cache.get(cache_key)

        if bundle is None or force:
            bundle = load_frame_bundle(
                sim_path=params['sim_path'],
                main_var=params['var_name'],
                t_range=(params['t_val'], params['t_val']),
                z_range=params['z_range'],
                x_range=params['x_range'],
                y_range=params['y_range'],
                wind_enabled=params['wind_enabled'],
                use_surface_wind=params['use_surface_wind'],
                contour_enabled=params['contour_enabled'],
                contour_var=params['contour_var'],
                use_cache=True,
                compute=True
            )
            self.cache.put(cache_key, bundle)

        return bundle

    def _prefetch_next_frame(self, params: Dict[str, Any]):
        """Prefetch next time frame asynchronously."""
        if not self.widgets['time_controls']['slider'].visible:
            return

        try:
            t_options = list(self.widgets['time_controls']['slider'].options.values())
            current_idx = self.widgets['time_controls']['slider'].value

            if current_idx in t_options:
                idx_pos = t_options.index(current_idx)
                if idx_pos < len(t_options) - 1:
                    next_pos = t_options[idx_pos + 1]
                    next_t = self.state.time_index_map.get(next_pos, next_pos)

                    req = FrameRequest(
                        sim_path=params['sim_path'],
                        var_name=params['var_name'],
                        t_range=(next_t, next_t),
                        z_range=params['z_range'],
                        x_range=params['x_range'],
                        y_range=params['y_range'],
                        wind_enabled=params['wind_enabled'],
                        use_surface=params['use_surface_wind'],
                        contour_enabled=params['contour_enabled'],
                        contour_var=params['contour_var']
                    )

                    def _loader(r):
                        return load_frame_bundle(
                            r.sim_path, r.var_name, r.t_range, r.z_range,
                            r.x_range, r.y_range,
                            wind_enabled=r.wind_enabled,
                            use_surface_wind=r.use_surface,
                            contour_enabled=r.contour_enabled,
                            contour_var=r.contour_var,
                            use_cache=True, compute=True
                        )

                    self.cache.prefetch_async(req, _loader)
        except Exception as e:
            logger.debug(f"Prefetch failed: {e}")

    def _update_clim_widgets(self, main_da):
        """Update color limit widgets from data if not locked."""
        if self.widgets['clim']['lock'].value:
            return

        try:
            new_vmin, new_vmax = calculate_color_limits(
                main_da,
                lock_clim=False,
                symmetric=self.widgets['clim']['symmetric'].value
            )

            if abs(self.widgets['clim']['vmin'].value - new_vmin) > 1e-6:
                self.widgets['clim']['vmin'].value = new_vmin

            if abs(self.widgets['clim']['vmax'].value - new_vmax) > 1e-6:
                self.widgets['clim']['vmax'].value = new_vmax
        except Exception as e:
            logger.warning(f"Failed to update clim widgets: {e}")

    def _prepare_overlays(self, bundle, params, main_ds, hover_dims):
        """Prepare all overlay layers."""
        overlays = []
        c_title = ""

        # Boundaries
        if self.widgets['overlays']['county'].value:
            overlays.append(get_county_boundaries())
        if self.widgets['overlays']['town'].value:
            overlays.append(get_town_boundaries())

        # Wind
        if 'wind' in bundle:
            hover_dims = self._process_wind_data(bundle, main_ds, hover_dims)

        # Contours
        if 'contour' in bundle:
            c_title, hover_dims = self._process_contour_data(bundle, main_ds, hover_dims, overlays)

        return overlays, c_title, hover_dims

    def _process_wind_data(self, bundle, main_ds, hover_dims):
        """Process wind vector data for hover info."""
        try:
            u, v = bundle['wind']
            u = squeeze_singleton_dims(u)
            v = squeeze_singleton_dims(v)

            wspd = np.sqrt(u**2 + v**2)
            angle_rad = np.arctan2(v, u)

            if self.widgets['overlays']['wind_hover'].value:
                wdir = (270 - angle_rad * 180 / np.pi) % 360
                main_ds['wind speed (m s-1)'] = wspd
                main_ds['wind direction (deg)'] = wdir
                hover_dims.extend(['wind speed (m s-1)', 'wind direction (deg)'])

        except Exception as e:
            logger.warning(f"Failed to process wind data: {e}")

        return hover_dims

    def _process_contour_data(self, bundle, main_ds, hover_dims, overlays):
        """Process contour overlay data."""
        c_title = ""
        try:
            c_da = squeeze_singleton_dims(bundle['contour'])

            if self.widgets['overlays']['contour'].value:
                # Determine contour range
                if self.state.auto_contour_range:
                    c_vmin, c_vmax = None, None
                    try:
                        self.state.updating_contour_programmatically = True
                        self.widgets['overlays']['contour_vmin'].value = float(c_da.min())
                        self.widgets['overlays']['contour_vmax'].value = float(c_da.max())
                    finally:
                        self.state.updating_contour_programmatically = False
                else:
                    c_vmin = self.widgets['overlays']['contour_vmin'].value
                    c_vmax = self.widgets['overlays']['contour_vmax'].value

                # Build title
                c_name = self.widgets['var_selectors']['contour_variable'].value
                c_long_name = c_da.attrs.get('long_name', c_name)
                c_units = c_da.attrs.get('units', '')

                if c_units and c_units != 'N/A':
                    c_title = f"Contour: {c_long_name} [{c_units}]"
                else:
                    c_title = f"Contour: {c_long_name}"

                # Create overlay
                contour_overlay = create_contour_overlay(
                    c_da,
                    num_levels=self.widgets['overlays']['contour_levels'].value,
                    vmin=c_vmin,
                    vmax=c_vmax,
                    cmap_name='jet'
                )
                overlays.append(contour_overlay)

            # Hover info
            if self.widgets['overlays']['contour_hover'].value:
                c_name = self.widgets['var_selectors']['contour_variable'].value
                c_long_name = c_da.attrs.get('long_name', c_name)
                c_units = c_da.attrs.get('units', '')

                if c_units and c_units != 'N/A':
                    c_hover_name = f"{c_long_name} ({c_units})"
                else:
                    c_hover_name = c_long_name

                main_ds[c_hover_name] = c_da
                hover_dims.append(c_hover_name)

        except Exception as e:
            logger.warning(f"Failed to process contour data: {e}")

        return c_title, hover_dims

    def _build_plot_title(self, main_ds, params, c_title):
        """Build multi-line plot title."""
        var_name = params['var_name']
        da = main_ds[var_name]

        long_name = da.attrs.get('long_name', var_name)
        long_name = long_name[0].upper() + long_name[1:] if long_name else var_name
        units = da.attrs.get('units', '')

        # Format time
        if 'time' in main_ds.coords:
            t_val_formatted = format_time_value(main_ds.coords['time'].values)
        else:
            t_val_formatted = str(params['t_val'])

        # Format height
        z_info = ""
        if 'lev' in main_ds.coords:
            z_val = f"{float(main_ds.coords['lev'].values):.0f} m"
            z_info = f"  |  Height: {z_val}"
        elif params['z_val'] and self.widgets['lev_slider'].visible:
            z_info = f"  |  Height: {params['z_val']} m"

        # Build title
        title = f"{long_name}"
        if units and units != 'N/A':
            title += f" [{units}]"

        if c_title:
            title += f"\n{c_title}"

        title += f"\nTime: {t_val_formatted}{z_info}"

        return title

    def _create_base_plot(self, main_ds, params, title, hover_dims):
        """Create the base plot without overlays."""
        return create_main_plot(
            main_ds,
            cmap_name=params['cmap_name'],
            reverse_cmap=params['reverse_cmap'],
            vmin=self.widgets['clim']['vmin'].value,
            vmax=self.widgets['clim']['vmax'].value,
            lock_clim=self.widgets['clim']['lock'].value,
            symmetric_clim=params['symmetric_clim'],
            scale=self.widgets['scale_selector'].value,
            title=title,
            overlays=None,
            x_range=self.state.saved_x_range,
            y_range=self.state.saved_y_range,
            hover_dims=hover_dims
        )

    def _compose_final_plot(self, base_plot, overlays, bundle, params):
        """Compose final plot with all overlays."""
        final_plot = base_plot

        # Add static overlays
        for overlay in overlays:
            if overlay is not None:
                final_plot = final_plot * overlay

        # Add wind DynamicMap if enabled
        if params['wind_enabled'] and 'wind' in bundle:
            wind_dmap = self._create_wind_dmap(bundle)
            if wind_dmap is not None:
                final_plot = final_plot * wind_dmap
                logger.info("[WIND] Added DynamicMap wind overlay")

        return final_plot

    def _create_wind_dmap(self, bundle):
        """Create DynamicMap for wind vectors."""
        try:
            u, v = bundle['wind']
            u = squeeze_singleton_dims(u)
            v = squeeze_singleton_dims(v)

            lon_dim = 'lon' if 'lon' in u.dims else u.dims[-1]
            lat_dim = 'lat' if 'lat' in u.dims else u.dims[0]

            base_x_range = (float(u[lon_dim].min()), float(u[lon_dim].max()))
            base_y_range = (float(u[lat_dim].min()), float(u[lat_dim].max()))

            u_data = u
            v_data = v
            arrow_scale = self.widgets['overlays']['arrow_scale'].value
            arrow_density = self.widgets['overlays']['arrow_density'].value
            state = self.state

            def quiver_callback(x_range, y_range):
                if x_range is None or y_range is None:
                    x_range = state.saved_x_range
                    y_range = state.saved_y_range

                if x_range is None or y_range is None:
                    x_range = base_x_range
                    y_range = base_y_range

                result = create_wind_vectors(
                    u_data, v_data,
                    x_range=x_range,
                    y_range=y_range,
                    arrow_density=arrow_density,
                    arrow_scale=arrow_scale
                )
                return result if result is not None else hv.VectorField([])

            return hv.DynamicMap(quiver_callback, streams=[range_stream])

        except Exception as e:
            logger.warning(f"Failed to create wind DynamicMap: {e}")
            return None

    def _trigger_range_event(self):
        """Trigger initial range event for DynamicMap."""
        try:
            if self.state.saved_x_range is not None or self.state.saved_y_range is not None:
                range_stream.event(
                    x_range=self.state.saved_x_range,
                    y_range=self.state.saved_y_range
                )
        except Exception as e:
            logger.warning(f"Failed to trigger range event: {e}")

    # =========================================================================
    # View Controls
    # =========================================================================

    def reset_view(self, event):
        """Reset plot view (zoom/pan) to default."""
        self.state.skip_range_extraction = True
        self.state.saved_x_range = None
        self.state.saved_y_range = None

        self.update_plot(force=True)

        self.state.skip_range_extraction = False

        if pn.state.notifications:
            pn.state.notifications.info("View reset")
        logger.info("View reset")

    def reset_contour_range(self, event):
        """Reset contour range to auto mode."""
        self.state.auto_contour_range = True
        self.widgets['overlays']['contour_levels'].param.trigger('value')

        if pn.state.notifications:
            pn.state.notifications.info("Contour range reset to auto")
        logger.info("Contour range reset to auto")

    def on_contour_var_change(self, event):
        """Handle contour variable change - reset to auto range."""
        self.state.auto_contour_range = True
        logger.debug(f"Contour variable changed to {event.new}, resetting range")
        self.update_plot(force=True)

    def on_contour_range_edit(self, event):
        """Handle manual contour range edit - disable auto mode."""
        if not self.state.updating_contour_programmatically:
            self.state.auto_contour_range = False

    # =========================================================================
    # Callback Attachment
    # =========================================================================

    def attach_callbacks(self):
        """
        Attach all widget callbacks.

        This method connects all widgets to their respective handler methods.
        It should be called once after the controller is initialized.
        """
        logger.info("Attaching callbacks...")

        # Plot update triggers
        update_triggers = [
            (self.widgets['time_controls']['slider'], 'value'),
            (self.widgets['lev_slider'], 'value'),
            (self.widgets['range']['x'], 'value'),
            (self.widgets['range']['y'], 'value'),
            (self.widgets['cmap_selector'], 'value'),
            (self.widgets['cmap_reverse'], 'value'),
            (self.widgets['scale_selector'], 'value'),
            (self.widgets['clim']['lock'], 'value'),
            (self.widgets['clim']['symmetric'], 'value'),
            (self.widgets['clim']['vmin'], 'value'),
            (self.widgets['clim']['vmax'], 'value'),
            (self.widgets['overlays']['wind'], 'value'),
            (self.widgets['overlays']['wind_hover'], 'value'),
            (self.widgets['overlays']['arrow_scale'], 'value_throttled'),
            (self.widgets['overlays']['arrow_density'], 'value_throttled'),
            (self.widgets['overlays']['contour'], 'value'),
            (self.widgets['overlays']['contour_hover'], 'value'),
            (self.widgets['overlays']['contour_levels'], 'value_throttled'),
            (self.widgets['overlays']['contour_vmin'], 'value'),
            (self.widgets['overlays']['contour_vmax'], 'value'),
            (self.widgets['overlays']['county'], 'value'),
            (self.widgets['overlays']['town'], 'value')
        ]

        for widget, param_name in update_triggers:
            widget.param.watch(self.update_plot, param_name)

        # Load simulations button
        self.widgets['load_btn'].on_click(self.load_simulations)

        # Simulation selector
        self.widgets['sim_selector'].param.watch(self.on_simulation_change, 'value')

        # Load Data button
        self.widgets['buttons']['load'].on_click(self.load_data)

        # Reset view button
        self.widgets['buttons']['reset'].on_click(self.reset_view)

        # Variable category watchers
        self.widgets['var_selectors']['category'].param.watch(self.on_category_change, 'value')

        if 'contour_category' in self.widgets['var_selectors']:
            self.widgets['var_selectors']['contour_category'].param.watch(
                self.on_contour_category_change, 'value'
            )

        # Contour controls
        self.widgets['overlays']['contour_reset'].on_click(self.reset_contour_range)
        self.widgets['var_selectors']['contour_variable'].param.watch(
            self.on_contour_var_change, 'value'
        )
        self.widgets['overlays']['contour_vmin'].param.watch(self.on_contour_range_edit, 'value')
        self.widgets['overlays']['contour_vmax'].param.watch(self.on_contour_range_edit, 'value')

        logger.info("All callbacks attached")
