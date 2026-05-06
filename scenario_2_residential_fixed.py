#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scenario 2 — Residential / core densification.

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
    py scenario_2_residential_fixed.py
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


def transform_residential(feat, _blocks):
    X = feat.copy()

    center = np.array([X["x"].mean(), X["y"].mean()], dtype=np.float32)
    core_gain = np.exp(
        -(((X["x"] - center[0]) ** 2 + (X["y"] - center[1]) ** 2).to_numpy()
          / (2 * (2200.0 ** 2)))
    )

    # Residential development factors
    X["living_share"] = np.clip(X["living_share"] + 0.22 * core_gain, 0, 1)
    X["pop_density"] = X["pop_density"] + 500 * core_gain
    X["edu_count_dens"] = X["edu_count_dens"] + 0.25 * core_gain
    X["health_count_dens"] = X["health_count_dens"] + 0.20 * core_gain

    # Keep recreation as supporting environment, not dominant conversion
    X["recreation_count_dens"] = X["recreation_count_dens"] + 0.04 * core_gain

    # Accessibility also supports housing
    X["transport_count_dens"] = X["transport_count_dens"] + 0.12 * core_gain
    X["mean_drive_time"] = np.clip(X["mean_drive_time"] - 1.5 * core_gain, 0, None)

    return X.replace([np.inf, -np.inf], 0).fillna(0), core_gain


def constraint_residential(rows, class_to_idx):
    # Main target class: residential.
    if "LandUse.RESIDENTIAL" in class_to_idx:
        rows[:, class_to_idx["LandUse.RESIDENTIAL"]] *= 1.30

    # Moderate business growth as supporting urban service function.
    if "LandUse.BUSINESS" in class_to_idx:
        rows[:, class_to_idx["LandUse.BUSINESS"]] *= 1.08

    # Do not overinflate recreation in residential scenario.
    if "LandUse.RECREATION" in class_to_idx:
        rows[:, class_to_idx["LandUse.RECREATION"]] *= 0.80


def main() -> None:
    run_scenario(
        scenario_id="spb_residential_core",
        out_subdir="02_residential",
        transform_feat=transform_residential,
        constraint_fn=constraint_residential,
        base=ROOT,
    )


if __name__ == "__main__":
    main()
