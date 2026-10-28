import sys
from pathlib import Path

# Ensure src is on sys.path even when pytest is invoked without PYTHONPATH
src_path = Path(__file__).resolve().parents[1] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

