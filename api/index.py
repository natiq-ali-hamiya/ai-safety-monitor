"""
Vercel Serverless Function Entry Point for AI Safety Monitoring System.
Imports FastAPI app directly from main.py to maintain a single source of truth.
"""

from main import app  # noqa: F401
