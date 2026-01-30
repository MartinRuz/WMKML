#!/bin/bash

set -euo pipefail

PIDS=()

# cleanup processes when exiting the script
cleanup() {
    echo "Shutting down services..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait
    echo "Cleanup complete."
}


trap cleanup EXIT INT TERM

# Activate virtual environment
if [ -d "env" ]; then
    source env/bin/activate
else
    echo "Error: env/ not found."
    exit 1
fi

# Move Streamlit-config to the proper location
mkdir -p ~/.streamlit
cp config.toml ~/.streamlit/config.toml

# Start background services
echo "Starting Streamlit GUI"
streamlit run gui.py > logs/gui.log 2>&1 &
PIDS+=($!)

echo "Starting inference service"
python inference_owner.py > logs/inference.log 2>&1 &
PIDS+=($!)

echo "Starting owner service"
python owner.py > logs/owner.log 2>&1 &
PIDS+=($!)

HOST="0.0.0.0"
PORT=8000

echo "Waiting for service on port $PORT..."
until curl -s "http://localhost:$PORT/generate_gui" >/dev/null; do
  sleep 1
done

echo "Service is up!"
python trainer.py > logs/trainer.log 2>&1 &
PIDS+=($!)

echo "All services started. Press Ctrl+C to stop."

# Block forever (or until a child exits)
wait