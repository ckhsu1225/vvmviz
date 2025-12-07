"""
VVMViz UI Module

This module provides the user interface components for VVMViz, including:
- Widget factories for all Panel controls
- Interactive domain map selector
- Playback controller for time animation
- Dashboard layout assembly

The UI module is organized into submodules:
- widgets: Factory functions for creating Panel widgets
- map_selector: Interactive terrain map for domain selection
- playback: Time animation playback controller
- layout: Dashboard layout assembly functions

Examples
--------
Create a complete dashboard:

>>> from vvmviz.ui import create_dashboard
>>> from vvmviz.core.data_loader import scan_variable_groups
>>>
>>> # Scan available variables
>>> groups = scan_variable_groups('/data2/VVM/sim001/')
>>>
>>> # Create dashboard
>>> dashboard = create_dashboard(groups)
>>>
>>> # Serve
>>> import panel as pn
>>> pn.serve(dashboard, port=5006)

Create individual components:

>>> from vvmviz.ui.widgets import create_path_selector, create_range_sliders
>>> from vvmviz.ui.map_selector import DomainMapSelector
>>> from vvmviz.ui.playback import PlaybackController
>>>
>>> # Create widgets
>>> path_input, load_btn = create_path_selector()
>>> sliders = create_range_sliders()
>>>
>>> # Create map selector
>>> map_selector = DomainMapSelector(sliders['x'], sliders['y'])
>>>
>>> # Create playback controller
>>> time_controls = create_time_controls()
>>> playback = PlaybackController(
...     time_slider=time_controls['slider'],
...     play_button=time_controls['play']
... )
"""

# Widget factories
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

# Interactive components
from vvmviz.ui.map_selector import DomainMapSelector
from vvmviz.ui.playback import PlaybackController, create_playback_controller

# Layout assembly
from vvmviz.ui.layout import (
    create_dashboard,
    create_simple_dashboard,
    create_all_widgets,
    create_sidebar,
    create_main_area
)

__all__ = [
    # Widget factories
    'create_path_selector',
    'create_simulation_selector',
    'create_range_sliders',
    'create_variable_selectors',
    'create_colormap_gallery',
    'create_scale_selector',
    'create_colorbar_range_controls',
    'create_overlay_controls',
    'create_time_controls',
    'create_level_control',
    'create_action_buttons',
    'create_metadata_pane',
    'create_plot_pane',

    # Interactive components
    'DomainMapSelector',
    'PlaybackController',
    'create_playback_controller',

    # Layout assembly
    'create_dashboard',
    'create_simple_dashboard',
    'create_all_widgets',
    'create_sidebar',
    'create_main_area'
]
