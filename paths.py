import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))  # This is your Project Root
scratch_path = Path("/scratch")
if scratch_path.exists() and scratch_path.is_dir():
    DATA_DIR = scratch_path
else:
    DATA_DIR = ROOT_DIR / "data"
