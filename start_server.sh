#!/bin/bash
# VVMViz Dashboard Startup Script
# Usage: ./start_server.sh

# Kill any existing process on port 5006
PID=$(lsof -t -i:5006 2>/dev/null)
if [ -n "$PID" ]; then
    echo "Killing existing process on port 5006 (PID: $PID)"
    kill $PID
    sleep 1
fi

cd /data/ckhsu/vvmviz
source .venv/bin/activate
panel serve app.py --port 5006 --allow-websocket-origin="*"
