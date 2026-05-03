from __future__ import annotations
from __future__ import annotations
from pathlib import Path
import math
import pickle
import random
import re
from typing import Dict, Any, List, Tuple
import numpy as np
VEHICLE_SPEED = 10.0
DRONE_SPEED = 20.0
DRONE_MAX_RANGE = 10000.0
CAR_DRONE_MAX_RANGE = 10000.0
DRONE_PAYLOAD = 30.0
CAR_DRONE_PAYLOAD = 30.0
FACILITY_NOISE_THRESHOLD_DB = 40.0
DRONE_LW = 86.6
ALPHA = 0.005
SCALE = 100.0

def parse_solomon_txt(file_path: Path, n_customers: int) -> Dict[str, Any]:
    rows = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            nums = re.findall('-?\\d+', line)
            if len(nums) >= 7:
                vals = list(map(int, nums[:7]))
                rows.append(vals)
    rows = sorted(rows, key=lambda r: r[0])
    if not rows:
        raise ValueError(f'No Solomon customer rows found in {file_path}')
    depot = rows[0]
    customers = rows[1:1 + n_customers]
    if len(customers) < n_customers:
        raise ValueError(f'Requested N={n_customers}, but only {len(customers)} customers were found in {file_path}')
    warehouse_position = (depot[1] * SCALE, depot[2] * SCALE)
    customer_positions = [(r[1] * SCALE, r[2] * SCALE) for r in customers]
    weights = [0.0] + [float(r[3]) for r in customers]
    time_windows = [(0.0, float(depot[5]) * 10.0)] + [(0.0, float(r[5]) * 10.0) for r in customers]
    service_times = [0.0] + [float(r[6]) for r in customers]
    return {'warehouse_position': warehouse_position, 'customer_positions': customer_positions, 'weights': weights, 'time_windows': time_windows, 'service_times': service_times}

def build_distance_matrices(warehouse_position, customer_positions):
    nodes = [warehouse_position] + list(customer_positions)
    arr = np.asarray(nodes, dtype=float)
    diff = arr[:, None, :] - arr[None, :, :]
    manhattan = np.abs(diff).sum(axis=-1)
    euclidean = np.sqrt((diff ** 2).sum(axis=-1))
    return (manhattan, euclidean)

def attenuated_noise(customer_pos, facility_pos):
    d = math.dist(customer_pos, facility_pos)
    d = max(d, 1.0)
    return DRONE_LW - 20.0 * math.log10(d) - ALPHA * d

def build_facilities(seed: int, count: int=100):
    random.seed(seed)
    return [(random.uniform(0, 100 * SCALE), random.uniform(0, 100 * SCALE)) for _ in range(count)]

def build_static_inputs(instance_file: str, n_customers: int, solomon_dir: Path, facility_seed: int=42) -> Dict[str, Any]:
    parsed = parse_solomon_txt(Path(solomon_dir) / instance_file, n_customers)
    warehouse_position = parsed['warehouse_position']
    customer_positions = parsed['customer_positions']
    weights = parsed['weights']
    time_windows = parsed['time_windows']
    manhattan, euclidean = build_distance_matrices(warehouse_position, customer_positions)
    theta_tr = manhattan / VEHICLE_SPEED
    theta_d = euclidean / DRONE_SPEED
    N = n_customers
    facilities = build_facilities(facility_seed)
    car_range_ok = np.zeros((N + 1, N + 1, N + 1), dtype=bool)
    recovery_candidates = {}
    nodes = range(N + 1)
    for i in nodes:
        for j in range(1, N + 1):
            if i == j:
                continue
            cand = [p for p in nodes if p != i and p != j]
            recovery_candidates[i, j] = cand
            for p in cand:
                car_range_ok[i, j, p] = euclidean[i, j] + euclidean[j, p] <= CAR_DRONE_MAX_RANGE + 1e-09
    customer_allowed = []
    direct_ok = {}
    car_ok = {}
    direct_range_ok = {}
    for j in range(1, N + 1):
        pos = customer_positions[j - 1]
        noise_ok = all((attenuated_noise(pos, fpos) <= FACILITY_NOISE_THRESHOLD_DB for fpos in facilities))
        direct_range = 2.0 * euclidean[0, j] <= DRONE_MAX_RANGE + 1e-09
        direct_range_ok[j] = int(direct_range)
        direct_ok[j] = int(weights[j] <= DRONE_PAYLOAD and noise_ok and direct_range)
        car_feasible = bool(np.any(car_range_ok[:, j, :]))
        car_ok[j] = int(weights[j] <= CAR_DRONE_PAYLOAD and noise_ok and car_feasible)
        allow = [0]
        if car_ok[j]:
            allow.append(2)
        if direct_ok[j]:
            allow.append(1)
        customer_allowed.append(allow)
    return {'instance_file': instance_file, 'N': N, 'warehouse_position': warehouse_position, 'customer_positions': customer_positions, 'weights': weights, 'time_windows': time_windows, 'service_times': parsed['service_times'], 'facility_positions': facilities, 'manhattan_matrix': manhattan, 'euclidean_matrix': euclidean, 'theta_tr': theta_tr, 'theta_d': theta_d, 'direct_ok': direct_ok, 'car_ok': car_ok, 'direct_range_ok': direct_range_ok, 'car_range_ok_matrix': car_range_ok, 'recovery_candidates': recovery_candidates, 'customer_allowed': customer_allowed, 'num_car_drones': 1}

def ensure_precomputed(instance_file: str, N: int, solomon_dir: Path, output_dir: Path, facility_seed: int=42, rebuild: bool=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(instance_file).stem
    pkl_path = output_dir / f'{tag}_N{N}_seed{facility_seed}.pkl'
    if pkl_path.exists() and (not rebuild):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    pre = build_static_inputs(instance_file, N, Path(solomon_dir), facility_seed)
    with open(pkl_path, 'wb') as f:
        pickle.dump(pre, f)
    return pre
