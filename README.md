# VVMViz - Interactive VVM Data Visualization

A modular, interactive dashboard for visualizing Vector Vorticity Model (VVM) simulation data.

## Features

- Interactive Visualization: Real-time exploration of VVM simulation data
- NCL Colormaps: 37 professional atmospheric science colormaps
- Multiple Overlays: Wind vectors, contours, and Taiwan boundaries
- Performance: Two-layer caching with background prefetching
- Playback Control: Time animation with adjustable speed
- Domain Selection: Interactive terrain map for spatial range selection
- Flexible Plotting: Linear/log scaling, symmetric/locked color limits

## Installation

### Requirements

- Python 3.13+
- [VVM Reader](https://github.com/ckhsu1225/vvm_reader) library
- Panel, HoloViews, Xarray, Dask

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ckhsu1225/vvmviz.git
cd vvmviz
```

2. Create virtual environment with uv:
```bash
uv venv
source .venv/bin/activate
uv sync
```

## Quick Start

### Launch Dashboard

```bash
source .venv/bin/activate
panel serve app.py --show --port 5006
```

The dashboard will open at http://localhost:5006

### Run as Background Service

```bash
# Start in background with screen
screen -dmS vvmviz ./start_server.sh

# Check status
screen -ls

# View logs
screen -r vvmviz
# (Press Ctrl+A then D to detach)

# Stop service
screen -X -S vvmviz quit
```

## Project Structure

```
vvmviz/
├── app.py                      # Main dashboard entry point
├── start_server.sh             # Background server startup script
├── pyproject.toml              # Project dependencies (uv)
├── LICENSE                     # MIT License
└── src/vvmviz/                 # Source code
    ├── config.py               # Configuration and constants
    ├── state.py                # Application state management
    │
    ├── controllers/            # Controller layer (MVC pattern)
    │   └── app_controller.py   # Main application controller
    │
    ├── core/                   # Data processing
    │   ├── data_loader.py      # VVM data loading
    │   └── data_processor.py   # Data processing & slicing
    │
    ├── ui/                     # User interface
    │   ├── widgets.py          # Panel widget factories
    │   ├── map_selector.py     # Interactive domain map
    │   ├── playback.py         # Playback controller
    │   └── layout.py           # Dashboard layout
    │
    ├── plotting/               # Visualization
    │   ├── base.py             # Core plotting functions
    │   ├── colormaps.py        # NCL colormap management
    │   └── overlays.py         # Wind, contour, boundaries
    │
    └── utils/                  # Utilities
        ├── cache.py            # Cache manager (LRU + prefetch)
        ├── metadata.py         # Metadata formatting
        └── shapefile.py        # Taiwan boundary shapefiles
```

## Architecture

VVMViz follows the Model-View-Controller (MVC) pattern:

- Model: core/ - Data loading and processing
- View: ui/ - Widgets and layout
- Controller: controllers/ - Application logic and callbacks
- State: state.py - Centralized application state

## Configuration

Default configuration in config.py:

```python
DEFAULT_VVM_DIR = '/data2/VVM/taiwanvvm_summer/'
TWCOUNTY_SHP_PATH = '/path/to/county.shp'
TWTOWN_SHP_PATH = '/path/to/town.shp'
MAX_FRAME_CACHE_SIZE = 200
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [VVM Reader](https://github.com/ckhsu1225/vvm_reader) for data access
- [Panel](https://panel.holoviz.org/) / [HoloViews](https://holoviews.org/) for interactive visualization
- NCL colormaps for professional color schemes
- Taiwan MOI for administrative boundary shapefiles
