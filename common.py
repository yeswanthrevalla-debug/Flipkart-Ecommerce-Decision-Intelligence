"""Shared, side-effect-free helpers used by Part 2 and Part 4."""
import numpy as np


def stable_argsort(values):
    """Sort values stably so ties retain reference-row order."""
    return np.argsort(values, kind="stable")
