"""
UI Widgets Module

This module provides factory functions for creating all Panel widgets used in VVMViz.
Each function creates a properly configured widget with default values and settings.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import panel as pn
import matplotlib.pyplot as plt
import cmaps

from vvmviz.config import DEFAULT_VVM_DIR
from vvmviz.plotting.colormaps import NCL_CMAP_CATEGORIES, DEFAULT_COLORMAP, ALL_NCL_CMAPS

logger = logging.getLogger(__name__)


# =============================================================================
# Path and Simulation Selection
# =============================================================================

def create_path_selector(default_path: Optional[Path] = None) -> Tuple[pn.widgets.AutocompleteInput, pn.widgets.Button]:
    """
    Create path input widget with autocomplete and load button.

    Parameters
    ----------
    default_path : Path, optional
        Default directory path (default: from config)

    Returns
    -------
    tuple of (AutocompleteInput, Button)
        Path input widget and load button

    Examples
    --------
    >>> path_input, load_btn = create_path_selector()
    >>> path_input.value
    '/data2/VVM/taiwanvvm_summer/'
    """
    if default_path is None:
        default_path = DEFAULT_VVM_DIR

    path_input = pn.widgets.AutocompleteInput(
        name='Data Directory',
        value=str(default_path),
        placeholder='Enter path to VVM simulations...',
        min_characters=1,
        case_sensitive=False,
        restrict=False,
        search_strategy='starts_with'
    )

    load_btn = pn.widgets.Button(
        name='Load Simulations',
        button_type='primary',
        width=120
    )

    # Autocomplete suggestion handler
    def update_path_suggestions(event):
        """Update autocomplete options based on input."""
        val = event.new
        if not val:
            return

        # Determine search directory
        search_dir = val if val.endswith('/') else os.path.dirname(val)

        if os.path.isdir(search_dir):
            try:
                # List directories only
                subdirs = [
                    os.path.join(search_dir, d) + '/'
                    for d in os.listdir(search_dir)
                    if os.path.isdir(os.path.join(search_dir, d)) and not d.startswith('.')
                ]
                path_input.options = sorted(subdirs)
            except Exception as e:
                logger.debug(f"Error listing directory: {e}")

    path_input.param.watch(update_path_suggestions, 'value_input')

    return path_input, load_btn


def create_simulation_selector() -> pn.widgets.Select:
    """
    Create simulation selector widget.

    Returns
    -------
    pn.widgets.Select
        Simulation selector widget (initially empty)

    Examples
    --------
    >>> sim_selector = create_simulation_selector()
    >>> sim_selector.options = {'sim001': Path('/data/sim001')}
    """
    return pn.widgets.Select(
        name='Simulation',
        options={}
    )


# =============================================================================
# Range Sliders
# =============================================================================

def create_range_sliders() -> Dict[str, pn.widgets.Widget]:
    """
    Create all range slider widgets (time, height, x, y).

    Returns
    -------
    dict
        Dictionary with keys: 'time', 'lev', 'x', 'y'
        Values are slider widgets with default ranges

    Examples
    --------
    >>> sliders = create_range_sliders()
    >>> sliders['time'].value
    (0, 100)
    >>> sliders['lev'].value
    (0, 20000)
    """
    return {
        'time': pn.widgets.IntRangeSlider(
            name='Time Range (Steps)',
            start=0,
            end=100,
            value=(0, 100)
        ),
        'lev': pn.widgets.RangeSlider(
            name='Height Range (m)',
            start=0,
            end=20000,
            value=(0, 20000),
            step=100
        ),
        'x': pn.widgets.IntRangeSlider(
            name='X Range (Indices)',
            start=0,
            end=100,
            value=(0, 100)
        ),
        'y': pn.widgets.IntRangeSlider(
            name='Y Range (Indices)',
            start=0,
            end=100,
            value=(0, 100)
        )
    }


# =============================================================================
# Variable Selection
# =============================================================================

def create_variable_selectors(
    variable_groups: Dict[str, List[str]],
    include_contour: bool = True
) -> Dict[str, pn.widgets.Widget]:
    """
    Create variable selection widgets (category and variable).

    Parameters
    ----------
    variable_groups : dict
        Dictionary mapping category names to lists of variable names
    include_contour : bool, default=True
        If True, also create contour variable selectors

    Returns
    -------
    dict
        Dictionary with keys:
        - 'category': Category selector
        - 'variable': Variable selector
        - 'contour_category': Contour category selector (if include_contour)
        - 'contour_variable': Contour variable selector (if include_contour)

    Examples
    --------
    >>> groups = {'File: Output': ['qc', 'qr', 'th']}
    >>> selectors = create_variable_selectors(groups)
    >>> selectors['category'].value
    'File: Output'
    """
    if not variable_groups:
        raise ValueError("variable_groups cannot be empty")

    init_cat = list(variable_groups.keys())[0]
    init_var = variable_groups[init_cat][0]

    # Main variable selectors
    category_selector = pn.widgets.Select(
        name='Group Category',
        options=list(variable_groups.keys()),
        value=init_cat
    )

    variable_selector = pn.widgets.Select(
        name='Variable',
        options=variable_groups[init_cat],
        value=init_var
    )

    result = {
        'category': category_selector,
        'variable': variable_selector
    }

    # Contour variable selectors (if requested)
    if include_contour:
        contour_category_selector = pn.widgets.Select(
            name='Contour Category',
            options=list(variable_groups.keys()),
            value=init_cat
        )

        contour_variable_selector = pn.widgets.Select(
            name='Contour Variable',
            options=variable_groups[init_cat],
            value=init_var
        )

        result['contour_category'] = contour_category_selector
        result['contour_variable'] = contour_variable_selector

    return result


# =============================================================================
# Colormap Selection
# =============================================================================

def create_colormap_gallery() -> Tuple[pn.widgets.Select, pn.Column, pn.widgets.Checkbox]:
    """
    Create colormap selection widgets: hidden selector, visual gallery, and reverse checkbox.

    Returns
    -------
    tuple of (Select, Column, Checkbox)
        - cmap_selector: Hidden Select widget (holds current value)
        - cmap_gallery: Visual gallery of colormap buttons
        - reverse_checkbox: Reverse colormap checkbox

    Examples
    --------
    >>> selector, gallery, reverse_cb = create_colormap_gallery()
    >>> selector.value
    'WhiteBlueGreenYellowRed'
    >>> selector.visible
    False
    """
    # Hidden selector to hold value
    cmap_selector = pn.widgets.Select(
        name='Colormap',
        options=ALL_NCL_CMAPS,
        value=DEFAULT_COLORMAP,
        visible=False
    )

    # Visual gallery
    def create_cmap_button_row(cmap_name: str) -> pn.Row:
        """Create a button + preview row for a single colormap."""
        # Create matplotlib preview
        fig, ax = plt.subplots(figsize=(1.5, 0.2))
        fig.subplots_adjust(bottom=0, top=1, left=0, right=1)

        try:
            # Load colormap
            if hasattr(cmaps, cmap_name):
                cm = getattr(cmaps, cmap_name)
            else:
                cm = plt.get_cmap(cmap_name)

            cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cm), cax=ax, orientation='horizontal')
            cb.set_ticks([])
        except Exception as e:
            logger.warning(f"Failed to load colormap {cmap_name}: {e}")

        plt.close(fig)
        preview = pn.pane.Matplotlib(fig, tight=True, height=30)

        # Selection button
        btn = pn.widgets.Button(name=cmap_name, width=180, height=30, margin=(0, 5))

        def select_cmap(event):
            cmap_selector.value = cmap_name

        btn.on_click(select_cmap)

        return pn.Row(btn, preview)

    # Build gallery cards (categorized)
    gallery_cards = []
    for category, cmaps_list in NCL_CMAP_CATEGORIES.items():
        rows = [create_cmap_button_row(c) for c in cmaps_list]
        card = pn.Card(
            pn.Column(*rows, sizing_mode='stretch_width'),
            title=category,
            collapsed=True,
            sizing_mode='stretch_width',
            margin=(0, 0, 0, 20)
        )
        gallery_cards.append(card)

    gallery = pn.Column(*gallery_cards, height=300, scroll=True, css_classes=['cmap-gallery'])

    # Reverse checkbox
    reverse_checkbox = pn.widgets.Checkbox(name='Reverse Colormap', value=False)

    return cmap_selector, gallery, reverse_checkbox


# =============================================================================
# Color Scale and Range Controls
# =============================================================================

def create_scale_selector() -> pn.widgets.Select:
    """
    Create color scale selector (Linear/Log).

    Returns
    -------
    pn.widgets.Select
        Scale selector widget

    Examples
    --------
    >>> scale = create_scale_selector()
    >>> scale.value
    'Linear'
    """
    return pn.widgets.Select(
        name='Color Scale',
        options=['Linear', 'Log'],
        value='Linear',
        width=100
    )


def create_colorbar_range_controls() -> Dict[str, pn.widgets.Widget]:
    """
    Create colorbar range control widgets (vmin, vmax, lock, symmetric).

    Returns
    -------
    dict
        Dictionary with keys: 'vmin', 'vmax', 'lock', 'symmetric'

    Examples
    --------
    >>> controls = create_colorbar_range_controls()
    >>> controls['vmin'].value
    0.0
    >>> controls['lock'].value
    False
    """
    return {
        'vmin': pn.widgets.FloatInput(
            name='Min Value',
            value=0.0,
            step=0.1,
            width=100
        ),
        'vmax': pn.widgets.FloatInput(
            name='Max Value',
            value=10.0,
            step=0.1,
            width=100
        ),
        'lock': pn.widgets.Checkbox(
            name='Lock Range',
            value=False
        ),
        'symmetric': pn.widgets.Checkbox(
            name='Symmetric Range',
            value=False
        )
    }


# =============================================================================
# Overlay Controls
# =============================================================================

def create_overlay_controls() -> Dict[str, pn.widgets.Widget]:
    """
    Create overlay control widgets (wind, boundaries, contour).

    Returns
    -------
    dict
        Dictionary with overlay control widgets:
        - Wind overlay: 'wind', 'wind_hover', 'arrow_scale', 'arrow_density'
        - Boundaries: 'county', 'town'
        - Contour: 'contour', 'contour_levels', 'contour_vmin', 'contour_vmax',
                   'contour_reset', 'contour_hover'

    Examples
    --------
    >>> controls = create_overlay_controls()
    >>> controls['wind'].value
    False
    >>> controls['arrow_scale'].value
    1.0
    """
    return {
        # Wind overlay
        'wind': pn.widgets.Checkbox(
            name='Overlay Wind (u, v)',
            value=False
        ),
        'wind_hover': pn.widgets.Checkbox(
            name='Show Wind Info in Hover',
            value=False
        ),
        'arrow_scale': pn.widgets.FloatSlider(
            name='Arrow Scale',
            start=0.1,
            end=2.0,
            step=0.1,
            value=1.0
        ),
        'arrow_density': pn.widgets.IntSlider(
            name='Arrow Density',
            start=5,
            end=50,
            value=25
        ),

        # Boundary overlays
        'county': pn.widgets.Checkbox(
            name='Show County Boundaries',
            value=False
        ),
        'town': pn.widgets.Checkbox(
            name='Show Town Boundaries',
            value=False
        ),

        # Contour overlay
        'contour': pn.widgets.Checkbox(
            name='Overlay Contour',
            value=False
        ),
        'contour_levels': pn.widgets.IntSlider(
            name='Contour Levels',
            start=1,
            end=20,
            value=5
        ),
        'contour_vmin': pn.widgets.FloatInput(
            name='Contour Min',
            value=0.0,
            step=0.1,
            width=100
        ),
        'contour_vmax': pn.widgets.FloatInput(
            name='Contour Max',
            value=0.0,
            step=100,
            width=100
        ),
        'contour_reset': pn.widgets.Button(
            name='Reset Range',
            width=100,
            align='end'
        ),
        'contour_hover': pn.widgets.Checkbox(
            name='Show Contour Info in Hover',
            value=False
        )
    }


# =============================================================================
# Time and Level Controls
# =============================================================================

def create_time_controls() -> Dict[str, pn.widgets.Widget]:
    """
    Create time control widgets (slider, playback buttons, speed).

    Returns
    -------
    dict
        Dictionary with keys: 'slider', 'prev', 'next', 'play', 'speed'

    Examples
    --------
    >>> controls = create_time_controls()
    >>> controls['play'].name
    '▶'
    >>> controls['speed'].value
    1000
    """
    return {
        'slider': pn.widgets.DiscreteSlider(
            name='Time',
            options={'0': 0},
            value=0,
            visible=False
        ),
        'prev': pn.widgets.Button(
            name='⏮',
            width=40
        ),
        'next': pn.widgets.Button(
            name='⏭',
            width=40
        ),
        'play': pn.widgets.Button(
            name='▶',
            width=40
        ),
        'speed': pn.widgets.IntSlider(
            name='Speed (ms)',
            start=100,
            end=2000,
            step=100,
            value=1000,
            width=150,
            visible=False
        )
    }


def create_level_control() -> pn.widgets.DiscreteSlider:
    """
    Create vertical level control slider.

    Returns
    -------
    pn.widgets.DiscreteSlider
        Level slider (initially hidden)

    Examples
    --------
    >>> lev_slider = create_level_control()
    >>> lev_slider.visible
    False
    """
    return pn.widgets.DiscreteSlider(
        name='Height',
        options={'0 m': 0.0},
        value=0.0,
        visible=False,
        orientation='horizontal'
    )


# =============================================================================
# Action Buttons
# =============================================================================

def create_action_buttons() -> Dict[str, pn.widgets.Button]:
    """
    Create action buttons (Load Data, Reset View).

    Returns
    -------
    dict
        Dictionary with keys: 'load', 'reset'

    Examples
    --------
    >>> buttons = create_action_buttons()
    >>> buttons['load'].button_type
    'success'
    """
    return {
        'load': pn.widgets.Button(
            name='Load Data',
            button_type='success',
            width=120
        ),
        'reset': pn.widgets.Button(
            name='Reset View',
            button_type='warning',
            width=100
        )
    }


# =============================================================================
# Information Display
# =============================================================================

def create_metadata_pane() -> pn.pane.Markdown:
    """
    Create metadata display pane.

    Returns
    -------
    pn.pane.Markdown
        Markdown pane for displaying variable metadata

    Examples
    --------
    >>> metadata = create_metadata_pane()
    >>> metadata.object = "**Variable**: qc\\n**Units**: kg/kg"
    """
    return pn.pane.Markdown(
        "**Metadata**: Load data to see variable information.",
        sizing_mode='stretch_width'
    )


def create_plot_pane() -> pn.pane.HoloViews:
    """
    Create main plot display pane.

    Returns
    -------
    pn.pane.HoloViews
        HoloViews pane for displaying the main plot

    Examples
    --------
    >>> plot_pane = create_plot_pane()
    """
    return pn.pane.HoloViews(
        object=None
    )
