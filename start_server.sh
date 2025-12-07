#!/bin/bash
# VVMViz Dashboard Startup Script
# Usage: ./start_server.sh

cd /data/ckhsu/vvmviz
source .venv/bin/activate
panel serve app.py --port 5006 --allow-websocket-origin="*"
