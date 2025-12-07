"""
VVMViz Controllers Module

This module provides the controller layer for VVMViz, implementing the
Controller pattern to separate application logic from UI components.

The main controller (VVMVizController) orchestrates:
- Data loading and caching
- Plot updates and overlays
- Widget state synchronization
- User interaction callbacks

Example
-------
>>> from vvmviz.ui import create_dashboard
>>> from vvmviz.controllers import VVMVizController
>>>
>>> layout = create_dashboard(variable_groups={})
>>> controller = VVMVizController(
...     widgets=layout._vvmviz_widgets,
...     plot_pane=layout._vvmviz_plot_pane,
...     metadata_pane=layout._vvmviz_metadata_pane,
...     map_selector=layout._vvmviz_map_selector
... )
>>> controller.attach_callbacks()
"""

from vvmviz.controllers.app_controller import VVMVizController

__all__ = ['VVMVizController']
