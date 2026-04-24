"""
run.py — SoundMatch AI Application Entry Point
===============================================
Run with:  python run.py
"""

import os
import sys

# Ensure backend is in path
sys.path.insert(0, os.path.dirname(__file__))

from backend.app import app

if __name__ == "__main__":
    port    = int(os.environ.get("PORT", 5000))
    debug   = os.environ.get("DEBUG", "true").lower() == "true"
    host    = os.environ.get("HOST", "0.0.0.0")

    print("━" * 60)
    print("  🎵 SoundMatch AI — Starting Server")
    print(f"  📡  http://{host}:{port}")
    print(f"  🔧  Debug: {debug}")
    print("━" * 60)

    app.run(host=host, port=port, debug=debug)