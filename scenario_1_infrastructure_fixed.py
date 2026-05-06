#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scenario 1 — Infrastructure / synthetic transport ring.

FIXED VERSION
- No /mnt/data paths.
- Uses the folder where this script is located as data directory.
- Automatically checks/installs networkx before graph_drive.pickle is loaded.

Required files in the same folder:
    blocks.geojson
    graph_drive.pickle
    accessibility_matrix_drive.pickle
    gravity_scenario_core.py

Run on Windows:
    py scenario_1_infrastructure_fixed.py
"""


from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

def ensure_required_packages() -> None:
    """
    Make the scripts more Windows-friendly.

    The road graph is stored as a pickle created from a networkx graph.
    Even if the script itself does not explicitly use networkx, Python needs
    the networkx module to unpickle graph_drive.pickle.

    If automatic installation is not allowed in your environment, run manually:
        py -m pip install networkx
    """
    required = {
        "networkx": "networkx",
    }
    for module_name, pip_name in required.items():
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            print(f"[setup] Missing package: {module_name}. Trying to install {pip_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            importlib.import_module(module_name)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ensure_required_packages()

from gravity_scenario_core import run_scenario


import numpy as np


def transform_infrastructure(feat, _blocks):
    X = feat.copy()

    center = np.array([X["x"].mean(), X["y"].mean()], dtype=np.float32)
    r = np.sqrt((X["x"] - center[0]) ** 2 + (X["y"] - center[1]) ** 2).to_numpy()
    ring_radius = float(np.quantile(r, 0.62))
    ring_gain = np.exp(-((r - ring_radius) ** 2) / (2 * (1800.0 ** 2)))

    # Transport accessibility effect
    X["transport_count_dens"] = X["transport_count_dens"] + 1.20 * ring_gain
    X["reachable_20m"] = X["reachable_20m"] + 25 * ring_gain
    X["reachable_30m"] = X["reachable_30m"] + 40 * ring_gain
    X["mean_drive_time"] = np.clip(X["mean_drive_time"] - 3.0 * ring_gain, 0, None)

    # Secondary economic effects
    X["retail_food_count_dens"] = X["retail_food_count_dens"] + 0.30 * ring_gain
    X["industrial_count_dens"] = X["industrial_count_dens"] + 0.20 * ring_gain

    return X.replace([np.inf, -np.inf], 0).fillna(0), ring_gain


def constraint_infrastructure(rows, class_to_idx):
    # Scenario priority: transport + business.
    for target, multiplier in [
        ("LandUse.TRANSPORT", 1.25),
        ("LandUse.BUSINESS", 1.15),
    ]:
        if target in class_to_idx:
            rows[:, class_to_idx[target]] *= multiplier


def main() -> None:
    run_scenario(
        scenario_id="spb_infrastructure_ring",
        out_subdir="01_infrastructure",
        transform_feat=transform_infrastructure,
        constraint_fn=constraint_infrastructure,
        base=ROOT,
    )


if __name__ == "__main__":
    main()
