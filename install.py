from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    requirements = root / "requirements.txt"
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements)]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
