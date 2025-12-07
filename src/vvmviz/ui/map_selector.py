"""
Interactive Domain Map Selector

This module provides the DomainMapSelector class for interactive selection
of spatial domain ranges using a terrain map visualization.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import panel as pn
import holoviews as hv

from vvmviz.config import FILE_IO_LOCK
from vvmviz.core.data_loader import get_terrain_data

logger = logging.getLogger(__name__)


# =============================================================================
# Domain Map Selector Class
# =============================================================================

class DomainMapSelector:
    """
    Interactive map for selecting spatial domain range.

    This widget displays a terrain map and allows users to:
    1. Click two points on the map to define a rectangular selection
    2. See the current selection highlighted with a red box
    3. Sync selection with X/Y range sliders

    The map supports bidirectional synchronization:
    - Click on map → updates sliders
    - Change sliders → updates map box

    Attributes
    ----------
    terrain_plot : pn.pane.HoloViews
        HoloViews pane displaying the terrain map
    info_text : pn.pane.Markdown
        Markdown pane showing domain information and usage instructions
    tap_stream : hv.streams.Tap
        Bokeh tap stream for capturing click events
    box_stream : hv.streams.Pipe
        Pipe stream for updating selection box
    points_stream : hv.streams.Pipe
        Pipe stream for updating click markers
    xc_coords : np.ndarray
        X (longitude) coordinate array
    yc_coords : np.ndarray
        Y (latitude) coordinate array

    Parameters
    ----------
    x_range_slider : pn.widgets.IntRangeSlider
        X range slider widget to sync with
    y_range_slider : pn.widgets.IntRangeSlider
        Y range slider widget to sync with

    Examples
    --------
    >>> x_slider = pn.widgets.IntRangeSlider(name='X', start=0, end=256, value=(0, 256))
    >>> y_slider = pn.widgets.IntRangeSlider(name='Y', start=0, end=256, value=(0, 256))
    >>> selector = DomainMapSelector(x_slider, y_slider)
    >>> selector.create_terrain_map('/data2/VVM/sim001/')
    >>> panel_layout = selector.get_panel()
    """

    def __init__(self, x_range_slider, y_range_slider):
        """
        Initialize DomainMapSelector.

        Parameters
        ----------
        x_range_slider : pn.widgets.IntRangeSlider
            X range slider
        y_range_slider : pn.widgets.IntRangeSlider
            Y range slider
        """
        self.x_range_slider = x_range_slider
        self.y_range_slider = y_range_slider

        # UI components
        self.terrain_plot = pn.pane.HoloViews(
            object=None,
            sizing_mode='fixed',
            min_width=200
        )
        self.info_text = pn.pane.Markdown(
            "**Domain Selection**: Load a simulation to see the domain map."
        )

        # Streams (initialized in create_terrain_map)
        self.tap_stream: Optional[hv.streams.Tap] = None
        self.box_stream: Optional[hv.streams.Pipe] = None
        self.points_stream: Optional[hv.streams.Pipe] = None

        # Coordinates
        self.xc_coords: Optional[np.ndarray] = None
        self.yc_coords: Optional[np.ndarray] = None

        # State flags (prevent circular updates)
        self._updating_from_map = False
        self._updating_from_slider = False

        # Click tracking
        self.click_points = []  # Store two click points for rectangle

        # Watch slider changes
        self.x_range_slider.param.watch(self._on_slider_change, 'value')
        self.y_range_slider.param.watch(self._on_slider_change, 'value')

    def create_terrain_map(self, sim_path: str) -> bool:
        """
        Create terrain map visualization with interactive selection.

        This method:
        1. Loads terrain data for the simulation
        2. Creates HoloViews Image with terrain height
        3. Sets up tap stream for click events
        4. Creates dynamic selection box and click markers
        5. Updates info text with domain statistics

        Parameters
        ----------
        sim_path : str
            Path to VVM simulation directory

        Returns
        -------
        bool
            True if successful, False if error occurred

        Examples
        --------
        >>> selector = DomainMapSelector(x_slider, y_slider)
        >>> success = selector.create_terrain_map('/data2/VVM/sim001/')
        >>> if success:
        ...     print("Map created successfully")
        """
        try:
            # Load terrain data
            with FILE_IO_LOCK:
                terrain_da = get_terrain_data(sim_path)

            if terrain_da is None:
                logger.error("Failed to load terrain data")
                return False

            # Determine dimension names
            lon_dim = 'lon' if 'lon' in terrain_da.dims else list(terrain_da.dims)[-1]
            lat_dim = 'lat' if 'lat' in terrain_da.dims else list(terrain_da.dims)[0]

            # Store coordinate arrays
            self.xc_coords = terrain_da.coords[lon_dim].values
            self.yc_coords = terrain_da.coords[lat_dim].values

            # Create terrain image
            terrain_img = hv.Image(
                (self.xc_coords, self.yc_coords, terrain_da.values),
                kdims=['Longitude (°)', 'Latitude (°)'],
                vdims='Terrain Height (m)'
            ).opts(
                cmap='OceanLakeLandSnow',
                colorbar=True,
                frame_width=235,
                aspect='equal',
                tools=['tap', 'reset'],
                title='Domain Map',
                toolbar='below',
                fontsize={'title': '10pt', 'ticks': '8pt', 'labels': '8pt'},
                colorbar_opts={'width': 15, 'major_label_text_font_size': '8pt'},
            )

            # Create tap stream
            self.tap_stream = hv.streams.Tap(source=terrain_img, x=None, y=None)
            self.tap_stream.add_subscriber(self._on_tap)

            # Create pipe streams for manual updates
            self.box_stream = hv.streams.Pipe(data=[])
            self.points_stream = hv.streams.Pipe(data=[])

            # Create dynamic selection box
            def selection_box(data):
                """Draw selection box based on current slider values."""
                x_idx_range = self.x_range_slider.value
                y_idx_range = self.y_range_slider.value

                x_deg_range = self._indices_to_deg(x_idx_range, self.xc_coords)
                y_deg_range = self._indices_to_deg(y_idx_range, self.yc_coords)

                # Create rectangle: (x0, y0, x1, y1)
                rect_data = [(x_deg_range[0], y_deg_range[0], x_deg_range[1], y_deg_range[1])]

                return hv.Rectangles(rect_data).opts(
                    fill_alpha=0.1,
                    fill_color='red',
                    line_width=3,
                    line_color='red'
                )

            # Create dynamic click markers
            def click_markers(data):
                """Draw markers for clicked points."""
                if not self.click_points:
                    return hv.Points([]).opts(size=0)

                return hv.Points(self.click_points).opts(
                    color='yellow',
                    size=10,
                    marker='x',
                    line_width=2
                )

            # Create DynamicMaps
            box_dmap = hv.DynamicMap(selection_box, streams=[self.box_stream])
            points_dmap = hv.DynamicMap(click_markers, streams=[self.points_stream])

            # Overlay all layers
            combined = terrain_img * box_dmap * points_dmap

            # Trigger initial update
            self.box_stream.send([])
            self.points_stream.send([])

            # Update plot
            self.terrain_plot.object = combined

            # Update info text
            nx, ny = len(self.xc_coords), len(self.yc_coords)
            x_extent = (self.xc_coords.min(), self.xc_coords.max())
            y_extent = (self.yc_coords.min(), self.yc_coords.max())

            self.info_text.object = f"""
**Domain Info**:
- Grid size: {nx} × {ny}
- Longitude: {x_extent[0]:.2f}° - {x_extent[1]:.2f}°
- Latitude: {y_extent[0]:.2f}° - {y_extent[1]:.2f}°

**Usage**:
- Click two points on the map to define a rectangle
- First click: one corner, Second click: opposite corner
- The red box shows current selection
"""

            logger.info(f"Terrain map created for {sim_path}")
            return True

        except Exception as e:
            logger.error(f"Error creating terrain map: {e}", exc_info=True)
            self.info_text.object = f"**Error**: Failed to load terrain map: {e}"
            return False

    def _indices_to_deg(
        self,
        idx_range: Tuple[int, int],
        coords: np.ndarray
    ) -> Tuple[float, float]:
        """
        Convert index range to degree coordinates.

        Parameters
        ----------
        idx_range : tuple of int
            (start_idx, end_idx) index range
        coords : np.ndarray
            Coordinate array

        Returns
        -------
        tuple of float
            (min_coord, max_coord) in degrees
        """
        idx0, idx1 = idx_range

        # Clamp indices
        idx0 = max(0, min(idx0, len(coords) - 1))
        idx1 = max(idx0 + 1, min(idx1, len(coords)))

        coord0 = coords[idx0]
        coord1 = coords[idx1 - 1]

        return (float(coord0), float(coord1))

    def _deg_to_indices(
        self,
        deg_range: Tuple[float, float],
        coords: np.ndarray
    ) -> Tuple[int, int]:
        """
        Convert degree range to indices.

        Parameters
        ----------
        deg_range : tuple of float
            (min_deg, max_deg) degree range
        coords : np.ndarray
            Coordinate array

        Returns
        -------
        tuple of int
            (start_idx, end_idx) index range
        """
        deg0, deg1 = deg_range

        # Find nearest indices
        idx0 = int(np.argmin(np.abs(coords - deg0)))
        idx1 = int(np.argmin(np.abs(coords - deg1)))

        # Ensure idx1 > idx0
        if idx1 <= idx0:
            idx1 = idx0 + 1

        return (idx0, idx1)

    def _on_tap(self, x: Optional[float], y: Optional[float]):
        """
        Callback when user clicks on the map.

        Collects two click points and updates sliders when both are received.

        Parameters
        ----------
        x : float or None
            X coordinate (longitude) of click
        y : float or None
            Y coordinate (latitude) of click
        """
        if self._updating_from_slider:
            return

        if x is None or y is None:
            return

        if self.xc_coords is None or self.yc_coords is None:
            return

        try:
            # Add click point
            self.click_points.append((x, y))
            logger.debug(f"Click {len(self.click_points)}: ({x:.3f}°, {y:.3f}°)")

            # Update points display
            if self.points_stream is not None:
                self.points_stream.send([])

            # If we have two points, define the rectangle
            if len(self.click_points) == 2:
                self._updating_from_map = True

                p1, p2 = self.click_points

                # Define rectangle from two corner points
                x_range_deg = (min(p1[0], p2[0]), max(p1[0], p2[0]))
                y_range_deg = (min(p1[1], p2[1]), max(p1[1], p2[1]))

                # Convert degree ranges to indices
                x_indices = self._deg_to_indices(x_range_deg, self.xc_coords)
                y_indices = self._deg_to_indices(y_range_deg, self.yc_coords)

                # Update sliders
                self.x_range_slider.value = x_indices
                self.y_range_slider.value = y_indices

                logger.info(
                    f"Selected Lon: {x_range_deg[0]:.3f}°-{x_range_deg[1]:.3f}° → {x_indices}"
                )
                logger.info(
                    f"Selected Lat: {y_range_deg[0]:.3f}°-{y_range_deg[1]:.3f}° → {y_indices}"
                )

                # Update box display
                if self.box_stream is not None:
                    self.box_stream.send([])

                # Reset for next selection
                self.click_points = []
                self._updating_from_map = False

                # Clear points display
                if self.points_stream is not None:
                    self.points_stream.send([])

        except Exception as e:
            logger.error(f"Error in tap handler: {e}", exc_info=True)
            self.click_points = []
            if self.points_stream is not None:
                self.points_stream.send([])

    def _on_slider_change(self, event):
        """
        Callback when sliders change.

        Updates the selection box on the map to reflect new slider values.

        Parameters
        ----------
        event : param.Event
            Parameter change event
        """
        if self._updating_from_map:
            return

        if self.box_stream is None:
            return

        try:
            self._updating_from_slider = True
            # Trigger box redraw with new slider values
            self.box_stream.send([])
        except Exception as e:
            logger.error(f"Error updating map from sliders: {e}")
        finally:
            self._updating_from_slider = False

    def get_panel(self) -> pn.Column:
        """
        Get Panel layout for the domain map selector.

        Returns
        -------
        pn.Column
            Column layout containing info text and terrain plot

        Examples
        --------
        >>> selector = DomainMapSelector(x_slider, y_slider)
        >>> layout = selector.get_panel()
        >>> layout.servable()
        """
        # Set compact styling for info text
        self.info_text.margin = (-15, 0, 0, 10)
        
        return pn.Column(
            self.info_text,
            self.terrain_plot,
            sizing_mode='stretch_width',
        )
