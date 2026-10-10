"""Backend application package.

The repository root is added to ``sys.path`` here (once) so that the backend can
import the shared preprocessing module as ``ml.preprocess`` instead of copying it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
