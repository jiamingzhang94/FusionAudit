#!/usr/bin/env python3
"""Run the core FusionAudit rendered evaluation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fusionaudit.runner import main

if __name__ == "__main__":
    raise SystemExit(main("rendered"))
