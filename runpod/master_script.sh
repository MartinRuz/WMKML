#!/bin/bash

set -euo pipefail

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

echo "Starting inference service"
python inference_owner.py > logs/inference.log 2>&1 &

echo "Starting owner service"
python owner.py > logs/owner.log 2>&1 &

HOST="0.0.0.0"
PORT=8001

echo "Waiting for service on port $PORT..."
until curl -s "http://localhost:$PORT/docs" >/dev/null; do
  sleep 1
done

echo "Service is up!"

python trainer.py