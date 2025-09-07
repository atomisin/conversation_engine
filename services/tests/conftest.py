# services/conversation_engine/tests/conftest.py
import os
import sys
from pathlib import Path
# services/tests/conftest.py
import sys
import os

# Add src folder to PYTHONPATH
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # services/
SRC_DIR = os.path.join(BASE_DIR, "conversation_engine", "src")
sys.path.insert(0, SRC_DIR)
