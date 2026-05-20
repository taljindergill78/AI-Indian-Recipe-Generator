"""
HuggingFace Spaces entry point.

Spaces looks for app.py at the repo root by default.
This file simply delegates to the app package.

To run locally:
    uv run python app.py

To run in demo mode (no GPU required):
    INFERENCE_MODE=demo uv run python app.py
"""
from app.app import main

if __name__ == "__main__":
    main()
