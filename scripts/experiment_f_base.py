"""Baseline check of the F1-F3 harness: v1.9 as is, selection window, must equal the E1 record (alpha 0.2 / blend 0.75)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_common as xc

if __name__ == '__main__':
    xc.main({'base': xc.Variant()}, (1, 2), 'f-base')
