"""
Dashboard Layout Module

This module provides functions for assembling the complete VVMViz dashboard
layout from individual widgets and components.
"""

import logging
from typing import Dict, Any

import panel as pn

from vvmviz.ui.widgets import (
    create_path_selector,
    create_simulation_selector,
    create_range_sliders,
    create_variable_selectors,
    create_colormap_gallery,
    create_scale_selector,
    create_colorbar_range_controls,
    create_overlay_controls,
    create_time_controls,
    create_level_control,
    create_action_buttons,
    create_metadata_pane,
    create_plot_pane
)
from vvmviz.ui.map_selector import DomainMapSelector
from vvmviz.ui.playback import create_playback_controller

logger = logging.getLogger(__name__)


# =============================================================================
# Layout Assembly
# =============================================================================

def create_sidebar(
    widgets: Dict[str, Any],
    domain_map_selector: DomainMapSelector,
    metadata_pane: pn.pane.Markdown
) -> pn.Column:
    """
    Create sidebar with control cards and metadata.

    The sidebar contains collapsible cards for:
    1. Data Selection (path, simulation, variable)
    2. Data Range (time, height, x, y, domain map)
    3. Plot Settings (colormap, scale, color limits)
    4. Layer Overlays (boundaries, wind, contour)
    5. Variable Information (metadata)

    Parameters
    ----------
    widgets : dict
        Dictionary of all widget instances
    domain_map_selector : DomainMapSelector
        Domain map selector instance
    metadata_pane : pn.pane.Markdown
        Metadata display pane

    Returns
    -------
    pn.Column
        Sidebar column layout

    Examples
    --------
    >>> widgets = create_all_widgets(variable_groups)
    >>> map_selector = DomainMapSelector(widgets['range']['x'], widgets['range']['y'])
    >>> metadata = create_metadata_pane()
    >>> sidebar = create_sidebar(widgets, map_selector, metadata)
    """
    # 1. Data Selection Card
    card_data_selection = pn.Card(
        widgets['path_input'],
        widgets['load_btn'],
        pn.layout.Divider(),
        widgets['sim_selector'],
        widgets['var_selectors']['category'],
        widgets['var_selectors']['variable'],
        widgets['buttons']['load'],
        title="Data Selection",
        collapsed=False,
        sizing_mode='stretch_width'
    )

    # 2. Data Range Card
    card_data_range = pn.Card(
        pn.Column(
            widgets['range']['time'],
            widgets['range']['lev'],
            widgets['range']['x'],
            widgets['range']['y'],
            margin=(10, 0, 5, 10)  # (top, right, bottom, left) margins for range sliders
        ),
        pn.layout.Divider(),
        pn.pane.Markdown("### Domain Map Selector"),
        domain_map_selector.get_panel(),
        title="Data Range",
        collapsed=True,
        sizing_mode='stretch_width'
    )

    # 3. Plot Settings Card
    clim_row = pn.Column(
        pn.Row(
            widgets['clim']['lock'],
            widgets['clim']['symmetric']
        ),
        pn.Row(
            widgets['clim']['vmin'],
            widgets['clim']['vmax']
        )
    )

    card_plot_settings = pn.Card(
        widgets['cmap_gallery'],
        widgets['scale_selector'],
        widgets['cmap_reverse'],
        pn.layout.Divider(),
        clim_row,
        title="Color Settings",
        collapsed=True,
        sizing_mode='stretch_width'
    )

    # 4. Layer Overlays Card
    row_boundaries = pn.Row(
        widgets['overlays']['county'],
        widgets['overlays']['town']
    )

    col_wind = pn.Column(
        pn.Row(
            widgets['overlays']['wind'],
            widgets['overlays']['wind_hover']
        ),
        widgets['overlays']['arrow_scale'],
        widgets['overlays']['arrow_density'],
        name="Wind Vectors"
    )

    col_contour = pn.Column(
        pn.Row(
            widgets['overlays']['contour'],
            widgets['overlays']['contour_hover']
        ),
        widgets['var_selectors']['contour_category'],
        widgets['var_selectors']['contour_variable'],
        widgets['overlays']['contour_levels'],
        pn.Row(
            widgets['overlays']['contour_vmin'],
            widgets['overlays']['contour_vmax'],
            widgets['overlays']['contour_reset']
        ),
        name="Contour Overlay"
    )

    card_overlays = pn.Card(
        pn.pane.Markdown("#### Taiwan Boundaries"),
        row_boundaries,
        pn.layout.Divider(),
        pn.pane.Markdown("#### Wind Vectors"),
        col_wind,
        pn.layout.Divider(),
        pn.pane.Markdown("#### Contour Overlay"),
        col_contour,
        title="Layer Overlays",
        collapsed=True,
        sizing_mode='stretch_width'
    )

    # 5. Variable Information Card
    card_variable_info = pn.Card(
        metadata_pane,
        title="Variable Information",
        collapsed=True,
        sizing_mode='stretch_width'
    )

    # Assemble sidebar
    sidebar = pn.Column(
        pn.pane.Markdown("### Controls"),
        card_data_selection,
        card_overlays,
        card_plot_settings,
        card_data_range,
        pn.layout.Divider(),
        card_variable_info,
        width=390,  # Fixed width to save space for plot
        sizing_mode='stretch_height',
        scroll=True
    )

    return sidebar


def create_main_area(
    widgets: Dict[str, Any],
    plot_pane: pn.pane.HoloViews
) -> pn.Column:
    """
    Create main area with plot and controls.

    The main area contains:
    - Plot display with loading spinner and reset button
    - Time slider and playback controls (footer)
    - Level slider

    Parameters
    ----------
    widgets : dict
        Dictionary of all widget instances
    plot_pane : pn.pane.HoloViews
        Main plot pane

    Returns
    -------
    pn.Column
        Main area column layout

    Examples
    --------
    >>> widgets = create_all_widgets(variable_groups)
    >>> plot = create_plot_pane()
    >>> main = create_main_area(widgets, plot)
    """
    # Loading spinner (bound to Panel's busy state)
    loading_spinner = pn.indicators.LoadingSpinner(
        value=False,
        width=30,
        height=30,
        align='center',
        color='primary'
    )
    try:
        loading_spinner.value = pn.state.param.busy
    except Exception:
        # If pn.state.param.busy not available, just show static spinner
        pass

    # Plot area with controls
    plot_area = pn.Row(
        pn.Row(
            plot_pane,
            sizing_mode='stretch_width'
        ),
        pn.Row(
            loading_spinner,
            widgets['buttons']['reset'],
            sizing_mode='stretch_width',
            margin=(6, 0, 0, 80)
        )
    )

    # Footer with time/level controls
    footer = pn.Column(
        pn.Row(
            widgets['time_controls']['slider'],
            widgets['time_controls']['speed'],
            widgets['time_controls']['prev'],
            widgets['time_controls']['play'],
            widgets['time_controls']['next'],
            sizing_mode='stretch_width',
            align='center'
        ),
        pn.Row(
            widgets['lev_slider'],
            sizing_mode='stretch_width',
            align='center'
        ),
        sizing_mode='stretch_width'
    )

    # Assemble main area
    main_area = pn.Column(
        plot_area,
        footer,
        margin=(0, 0, 0, 40),
        sizing_mode='stretch_width'
    )

    return main_area


def create_all_widgets(variable_groups: Dict[str, list]) -> Dict[str, Any]:
    """
    Create all widgets needed for the dashboard.

    This is a convenience function that creates all widgets and returns
    them in a structured dictionary.

    Parameters
    ----------
    variable_groups : dict
        Dictionary mapping category names to lists of variable names

    Returns
    -------
    dict
        Dictionary with all widget instances, organized by category:
        - 'path_input': Path input widget
        - 'load_btn': Load simulations button
        - 'sim_selector': Simulation selector
        - 'var_selectors': Variable selectors dict
        - 'range': Range sliders dict
        - 'cmap_selector': Hidden colormap selector
        - 'cmap_gallery': Colormap gallery
        - 'cmap_reverse': Reverse colormap checkbox
        - 'scale_selector': Scale selector
        - 'clim': Color limit controls dict
        - 'overlays': Overlay controls dict
        - 'time_controls': Time control widgets dict
        - 'lev_slider': Level slider
        - 'buttons': Action buttons dict

    Examples
    --------
    >>> groups = {'File: Output': ['qc', 'qr', 'th']}
    >>> widgets = create_all_widgets(groups)
    >>> widgets['path_input'].value
    '/data2/VVM/taiwanvvm_summer/'
    """
    # Create all widgets
    path_input, load_btn = create_path_selector()
    sim_selector = create_simulation_selector()
    var_selectors = create_variable_selectors(variable_groups, include_contour=True)
    range_sliders = create_range_sliders()
    cmap_selector, cmap_gallery, cmap_reverse = create_colormap_gallery()
    scale_selector = create_scale_selector()
    clim_controls = create_colorbar_range_controls()
    overlay_controls = create_overlay_controls()
    time_controls = create_time_controls()
    lev_slider = create_level_control()
    buttons = create_action_buttons()

    # Package into structured dict
    return {
        'path_input': path_input,
        'load_btn': load_btn,
        'sim_selector': sim_selector,
        'var_selectors': var_selectors,
        'range': range_sliders,
        'cmap_selector': cmap_selector,
        'cmap_gallery': cmap_gallery,
        'cmap_reverse': cmap_reverse,
        'scale_selector': scale_selector,
        'clim': clim_controls,
        'overlays': overlay_controls,
        'time_controls': time_controls,
        'lev_slider': lev_slider,
        'buttons': buttons
    }


def create_dashboard(
    variable_groups: Dict[str, list],
    title: str = "VVM Visualization Dashboard"
) -> pn.Row:
    """
    Create complete dashboard layout.

    This is the main entry point for creating the VVMViz dashboard.
    It assembles all components (widgets, callbacks, layout) into a
    servable Panel application.

    Parameters
    ----------
    variable_groups : dict
        Dictionary mapping category names to lists of variable names
    title : str, default='VVM Visualization Dashboard'
        Dashboard title

    Returns
    -------
    pn.Row
        Complete dashboard layout ready to serve

    Notes
    -----
    This function creates the layout structure but does NOT attach callbacks.
    Callbacks should be attached separately before serving.

    Examples
    --------
    >>> from vvmviz.core.data_loader import scan_variable_groups
    >>> groups = scan_variable_groups('/data2/VVM/sim001/')
    >>> dashboard = create_dashboard(groups)
    >>> dashboard.servable()  # For panel serve
    >>> # OR
    >>> pn.serve(dashboard, port=5006)
    """
    # Create all widgets
    widgets = create_all_widgets(variable_groups)

    # Create domain map selector
    domain_map_selector = DomainMapSelector(
        widgets['range']['x'],
        widgets['range']['y']
    )

    # Create display panes
    metadata_pane = create_metadata_pane()
    plot_pane = create_plot_pane()

    # Create playback controller
    playback_controller = create_playback_controller(
        time_slider=widgets['time_controls']['slider'],
        play_button=widgets['time_controls']['play'],
        prev_button=widgets['time_controls']['prev'],
        next_button=widgets['time_controls']['next'],
        speed_slider=widgets['time_controls']['speed']
    )

    # Assemble layout
    sidebar = create_sidebar(widgets, domain_map_selector, metadata_pane)
    main_area = create_main_area(widgets, plot_pane)

    # Create sidebar toggle button
    toggle_button = pn.widgets.Button(
        name='◀',  # Left arrow to indicate collapse
        button_type='light',
        width=30,
        height=40,
        margin=(10, 0, 0, 0),
        align='start'
    )

    # Create a column to hold the sidebar (for easy show/hide)
    sidebar_container = pn.Column(
        sidebar,
        width=390,
        sizing_mode='stretch_height'
    )

    # Toggle functionality
    def toggle_sidebar(event=None):
        if sidebar_container.visible:
            # Collapse sidebar
            sidebar_container.visible = False
            toggle_button.name = '▶'  # Right arrow to indicate expand
            toggle_button.button_type = 'primary'
        else:
            # Expand sidebar
            sidebar_container.visible = True
            toggle_button.name = '◀'  # Left arrow to indicate collapse
            toggle_button.button_type = 'light'

    toggle_button.on_click(toggle_sidebar)

    # Create a row with toggle button and main area
    content_area = pn.Row(
        pn.Column(
            toggle_button,
            sizing_mode='stretch_height',
            width=20
        ),
        main_area,
        sizing_mode='stretch_both'
    )

    # Final layout with sidebar and content
    layout = pn.Row(
        sidebar_container,
        content_area,
        sizing_mode='stretch_both'
    )

    logger.info("Dashboard layout created with collapsible sidebar")

    # Store references for callback attachment
    layout._vvmviz_widgets = widgets
    layout._vvmviz_map_selector = domain_map_selector
    layout._vvmviz_playback = playback_controller
    layout._vvmviz_metadata_pane = metadata_pane
    layout._vvmviz_plot_pane = plot_pane
    layout._vvmviz_toggle_button = toggle_button
    layout._vvmviz_sidebar_container = sidebar_container

    return layout


# =============================================================================
# Simple Layout (for testing)
# =============================================================================

def create_simple_dashboard(variable_groups: Dict[str, list]) -> pn.Column:
    """
    Create a simplified dashboard for testing purposes.

    This creates a minimal layout with just the essential controls
    and plot area, useful for debugging and development.

    Parameters
    ----------
    variable_groups : dict
        Dictionary mapping category names to lists of variable names

    Returns
    -------
    pn.Column
        Simple dashboard layout

    Examples
    --------
    >>> groups = {'Test': ['qc']}
    >>> simple = create_simple_dashboard(groups)
    >>> simple.servable()
    """
    widgets = create_all_widgets(variable_groups)
    plot_pane = create_plot_pane()

    # Minimal layout
    controls = pn.Column(
        widgets['path_input'],
        widgets['load_btn'],
        widgets['sim_selector'],
        widgets['var_selectors']['category'],
        widgets['var_selectors']['variable'],
        widgets['buttons']['load'],
        width=300
    )

    layout = pn.Row(
        controls,
        plot_pane,
        sizing_mode='stretch_both'
    )

    return layout
