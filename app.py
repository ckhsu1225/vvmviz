"""
VVMViz Dashboard Application

This is the main entry point for the VVMViz dashboard application.
It assembles all components (data loading, UI, plotting, callbacks) and
serves the interactive visualization dashboard.

Usage
-----
Run with panel serve:
    $ panel serve app.py --show --port 5006

Or run directly:
    $ python app.py

The dashboard will be available at http://localhost:5006
"""

import logging

import panel as pn
import holoviews as hv

# Suppress Bokeh warnings
from bokeh.core.validation import silence
from bokeh.core.validation.warnings import FIXED_SIZING_MODE
silence(FIXED_SIZING_MODE, True)

# Initialize extensions
hv.extension('bokeh')
hv.config.image_rtol = 1.0
pn.extension(notifications=True)

# Import VVMViz modules
from vvmviz.ui import create_dashboard
from vvmviz.controllers import VVMVizController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        #logging.FileHandler('vvmviz_debug.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Main Dashboard Assembly
# =============================================================================

def create_vvmviz_dashboard():
    """
    Create and configure the complete VVMViz dashboard.

    This function:
    1. Creates the UI layout with all widgets
    2. Initializes the Controller
    3. Attaches all callbacks
    4. Returns servable dashboard

    Returns
    -------
    pn.Row
        Complete dashboard layout ready for serving

    Example
    -------
    >>> dashboard = create_vvmviz_dashboard()
    >>> dashboard.servable(title="VVM Visualization Dashboard")
    """
    logger.info("Creating VVMViz dashboard...")

    # Initialize with empty state - data will be loaded when user selects simulation
    variable_groups = {"No Simulation Loaded": ["-"]}

    # Create dashboard layout using UI factory
    layout = create_dashboard(variable_groups)

    # Retrieve components from layout
    widgets = layout._vvmviz_widgets
    map_selector = layout._vvmviz_map_selector
    metadata_pane = layout._vvmviz_metadata_pane
    plot_pane = layout._vvmviz_plot_pane

    # Create and configure controller
    controller = VVMVizController(
        widgets=widgets,
        plot_pane=plot_pane,
        metadata_pane=metadata_pane,
        map_selector=map_selector
    )

    # Attach all callbacks
    controller.attach_callbacks()

    # Store references for external access
    layout._vvmviz_controller = controller
    layout._vvmviz_state = controller.state
    layout._vvmviz_cache = controller.cache

    logger.info("Dashboard created successfully")

    return layout


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point when running as script."""
    logger.info("Starting VVMViz Dashboard Application")

    dashboard = create_vvmviz_dashboard()

    logger.info("Serving dashboard on http://localhost:5006")
    pn.serve(
        dashboard,
        port=5006,
        title="VVM Visualization Dashboard",
        show=True,
        autoreload=False
    )


# For panel serve
dashboard = create_vvmviz_dashboard()
dashboard.servable(title="VVM Visualization Dashboard")


if __name__ == '__main__':
    main()
