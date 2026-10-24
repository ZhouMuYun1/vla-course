#!/usr/bin/env python3
"""Root-level evaluate entry point (spec 9.3).

Thin wrapper around scripts/evaluate_openvla_widowx.py.
Run the formal 10-episode Base/Final comparison via:

    python evaluate.py --checkpoint <path> --task <task> --episodes 10 ...
"""
from scripts.evaluate_openvla_widowx import main

if __name__ == "__main__":
    main()
