"""Entry: python -m viewer (see viewer/app.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import main  # noqa: E402

sys.exit(main())
