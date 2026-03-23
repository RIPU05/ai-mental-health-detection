from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo = Path(__file__).resolve().parent
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_on_path()

from ai_mh_detection.dashboard.app import main  # noqa: E402


if __name__ == "__main__":
    main()

