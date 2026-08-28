import sys
import os
from pathlib import Path

# Add project root directory to Python module search path
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from main import app
