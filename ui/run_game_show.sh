#!/bin/bash

# GeoGuesser Game Show Launcher

echo "🎮 Starting GeoGuesser Game Show..."
echo ""
echo "╔════════════════════════════════════════╗"
echo "║   🌍 GEOGUESSER GAME SHOW 🌍          ║"
echo "║   The Ultimate Location Challenge      ║"
echo "╚════════════════════════════════════════╝"
echo ""

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "⚠️  Flask not found. Installing..."
    pip install flask
fi

# Check if the package is installed
if ! python -c "import dl_geoguesser" 2>/dev/null; then
    echo "⚠️  Package not installed. Installing in development mode..."
    cd ..
    pip install -e .
    cd ui
fi

echo ""
echo "🚀 Launching game show..."
echo "📱 Open your browser to: http://localhost:5001"
echo ""
echo "💡 Press Ctrl+C to stop the server"
echo ""

# Change to ui directory and run the app
cd "$(dirname "$0")"
python app.py
