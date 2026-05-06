#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scenario 3 — Real KAD-2 corridor influence.

FIXED VERSION
- No /mnt/data paths.
- Uses the folder where this script is located as data directory.
- Automatically checks/installs networkx before graph_drive.pickle is loaded.
- Looks for KAD-2 GeoJSON files in the same folder.

Required files in the same folder:
    blocks.geojson
    graph_drive.pickle
    accessibility_matrix_drive.pickle
    gravity_scenario_core.py

Optional KAD-2 files:
    KAD_2.geojson
    КАД_2.geojson
    kad2.geojson
    KAD_2_развязки.geojson
    КАД_2_развязки.geojson
    kad2_junctions.geojson

Run on Windows:
    py scenario_3_kad2_real_case_fixed.py
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


import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiPoint
from shapely.ops import unary_union


def find_file(base: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        p = base / name
        if p.is_file():
            return p
    return None


def load_kad_geometries(blocks: gpd.GeoDataFrame, base: Path):
    line_path = find_file(base, ("KAD_2.geojson", "КАД_2.geojson", "kad2.geojson"))
    junction_path = find_file(
        base,
        (
            "KAD_2_развязки.geojson",
            "КАД_2_развязки.geojson",
            "KAD_2_junctions.geojson",
            "kad2_junctions.geojson",
        ),
    )

    if line_path is not None:
        kad2 = gpd.read_file(line_path)
    else:
        bounds = blocks.total_bounds
        y_mid = float((bounds[1] + bounds[3]) / 2.0)
        kad2 = gpd.GeoDataFrame(
            geometry=[LineString([(bounds[0], y_mid), (bounds[2], y_mid)])],
            crs=blocks.crs,
        )
        print("[KAD-2] No corridor GeoJSON found. Using synthetic bbox corridor.")

    if junction_path is not None:
        kad2_j = gpd.read_file(junction_path)
    else:
        bounds = blocks.total_bounds
        y_mid = float((bounds[1] + bounds[3]) / 2.0)
        x0, x1 = float(bounds[0]), float(bounds[2])
        kad2_j = gpd.GeoDataFrame(
            geometry=[
                MultiPoint([
                    (x0 + 0.25 * (x1 - x0), y_mid),
                    (x0 + 0.75 * (x1 - x0), y_mid),
                ])
            ],
            crs=blocks.crs,
        )
        print("[KAD-2] No junction GeoJSON found. Using synthetic bbox junctions.")

    if blocks.crs is not None:
        if kad2.crs != blocks.crs:
            kad2 = kad2.to_crs(blocks.crs)
        if kad2_j.crs != blocks.crs:
            kad2_j = kad2_j.to_crs(blocks.crs)

    return unary_union(kad2.geometry.values), unary_union(kad2_j.geometry.values)


def transform_kad2(feat, blocks):
    kad2_line, kad2_junctions = load_kad_geometries(blocks, ROOT)

    centroids = blocks.geometry.centroid
    dist_route = centroids.distance(kad2_line).astype(np.float32)
    dist_junction = centroids.distance(kad2_junctions).astype(np.float32)

    route_gain = np.exp(-(dist_route.to_numpy() / 2500.0))
    junction_gain = np.exp(-(dist_junction.to_numpy() / 1500.0))
    kad2_gain = np.clip(0.72 * route_gain + 0.28 * junction_gain, 0, 1)

    X = feat.copy()

    # Direct infrastructure effect
    X["transport_count_dens"] = X["transport_count_dens"] + 1.45 * kad2_gain
    X["reachable_20m"] = X["reachable_20m"] + 30 * kad2_gain
    X["reachable_30m"] = X["reachable_30m"] + 46 * kad2_gain
    X["mean_drive_time"] = np.clip(X["mean_drive_time"] - 3.3 * kad2_gain, 0, None)

    # KAD-2 must generate business AND residential growth, not only transport.
    X["retail_food_count_dens"] = X["retail_food_count_dens"] + 0.42 * kad2_gain
    X["non_living_share"] = np.clip(X["non_living_share"] + 0.08 * kad2_gain, 0, 1)
    X["living_share"] = np.clip(X["living_share"] + 0.10 * kad2_gain, 0, 1)
    X["pop_density"] = X["pop_density"] + 180 * kad2_gain

    # Moderate industrial/logistic impact
    X["industrial_count_dens"] = X["industrial_count_dens"] + 0.18 * kad2_gain

    return X.replace([np.inf, -np.inf], 0).fillna(0), kad2_gain


def constraint_kad2(rows, class_to_idx):
    # KAD-2 scenario: transport + business + residential + moderate industry.
    for target, multiplier in [
        ("LandUse.TRANSPORT", 1.35),
        ("LandUse.BUSINESS", 1.25),
        ("LandUse.RESIDENTIAL", 1.22),
        ("LandUse.INDUSTRIAL", 1.10),
    ]:
        if target in class_to_idx:
            rows[:, class_to_idx[target]] *= multiplier

    # Do not allow recreation to dominate the infrastructure scenario.
    if "LandUse.RECREATION" in class_to_idx:
        rows[:, class_to_idx["LandUse.RECREATION"]] *= 0.75


def main() -> None:
    run_scenario(
        scenario_id="spb_kad2_corridor",
        out_subdir="03_kad2_corridor",
        transform_feat=transform_kad2,
        constraint_fn=constraint_kad2,
        base=ROOT,
    )


if __name__ == "__main__":
    main()
