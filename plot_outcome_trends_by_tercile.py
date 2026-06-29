"""Compatibility wrapper for the current co-primary tercile figures.

This file previously used the obsolete ``exposure_composite_hybrid`` measure
and an independent tercile-assignment routine. It now delegates to
``make_updated_thesis_figures.py``, which uses the exact regional assignments
from the preferred grouped DiD for both co-primary hybrid exposure measures.
"""
from make_updated_thesis_figures import main


if __name__ == "__main__":
    main()
