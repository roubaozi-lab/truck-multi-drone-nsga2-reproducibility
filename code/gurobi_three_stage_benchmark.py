from __future__ import annotations
import csv
import math
import os
import random
import re
import time
from pathlib import Path
from precompute_inputs import ensure_precomputed
import numpy as np
import gurobipy as gp
from gurobipy import GRB
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
BASE_DIR = Path(__file__).resolve().parent
SOLOMON_DIR = BASE_DIR / 'solomon_data'
INSTANCE_FILES = ['c101.txt', 'r101.txt', 'rc101.txt']
TEST_SCALES = [2, 4, 6, 8, 10]
TIME_LIMIT = 3600.0
STOP_AFTER_TIMELIMIT_AT_N = 10
V_TRUCK = 10.0
V_DRONE = 20.0
SERVICE_TIME = 10.0
SCALE = 100.0
TRUCK_UNIT_COST = 1.5 / 7.3 / 1000.0
DRONE_UNIT_COST = 0.5 / 7.3 / 1000.0
CAR_DRONE_UNIT_COST = 0.5 / 7.3 / 1000.0
WAIT_COST_PER_SEC = CAR_DRONE_UNIT_COST * V_DRONE
FIXED_TRUCK_COST = 100.0 / 7.3
FIXED_DIRECT_DRONE_COST = 20.0 / 7.3
FIXED_CAR_DRONE_COST = 20.0 / 7.3
DRONE_PAYLOAD = 30
CAR_DRONE_PAYLOAD = 30
DRONE_MAX_RANGE = 10000.0
CAR_DRONE_MAX_RANGE = 10000.0
DRONE_LW = 86.6
ALPHA = 0.005
FACILITY_NOISE_THRESHOLD_DB = 40.0
BIG_M = 100000.0
FALLBACK_WAREHOUSE = (35 * 100, 35 * 100)
FALLBACK_POSITIONS = [(41 * 100, 49 * 100), (35 * 100, 17 * 100), (55 * 100, 45 * 100), (55 * 100, 20 * 100), (15 * 100, 30 * 100), (25 * 100, 30 * 100), (20 * 100, 50 * 100), (10 * 100, 43 * 100), (55 * 100, 60 * 100), (30 * 100, 60 * 100), (20 * 100, 65 * 100), (50 * 100, 35 * 100), (30 * 100, 25 * 100), (15 * 100, 10 * 100), (30 * 100, 5 * 100), (10 * 100, 20 * 100), (5 * 100, 30 * 100), (20 * 100, 40 * 100), (15 * 100, 60 * 100), (45 * 100, 65 * 100), (45 * 100, 20 * 100), (45 * 100, 10 * 100), (55 * 100, 5 * 100), (65 * 100, 35 * 100), (65 * 100, 20 * 100), (45 * 100, 30 * 100), (35 * 100, 40 * 100), (41 * 100, 37 * 100), (64 * 100, 42 * 100), (40 * 100, 60 * 100), (31 * 100, 52 * 100), (35 * 100, 69 * 100), (53 * 100, 52 * 100), (65 * 100, 55 * 100), (63 * 100, 65 * 100), (2 * 100, 60 * 100), (20 * 100, 20 * 100), (5 * 100, 5 * 100), (60 * 100, 12 * 100), (40 * 100, 25 * 100), (42 * 100, 7 * 100), (24 * 100, 12 * 100), (23 * 100, 3 * 100), (11 * 100, 14 * 100), (6 * 100, 38 * 100), (2 * 100, 48 * 100), (8 * 100, 56 * 100), (13 * 100, 52 * 100), (6 * 100, 68 * 100), (47 * 100, 47 * 100), (49 * 100, 58 * 100), (27 * 100, 43 * 100), (37 * 100, 31 * 100), (57 * 100, 29 * 100), (63 * 100, 23 * 100), (53 * 100, 12 * 100), (32 * 100, 12 * 100), (36 * 100, 26 * 100), (21 * 100, 24 * 100), (17 * 100, 34 * 100), (12 * 100, 24 * 100), (24 * 100, 58 * 100), (27 * 100, 69 * 100), (15 * 100, 77 * 100), (62 * 100, 77 * 100), (49 * 100, 73 * 100), (67 * 100, 5 * 100), (56 * 100, 39 * 100), (37 * 100, 47 * 100), (37 * 100, 56 * 100), (57 * 100, 68 * 100), (47 * 100, 16 * 100), (44 * 100, 17 * 100), (46 * 100, 13 * 100), (49 * 100, 11 * 100), (49 * 100, 42 * 100), (53 * 100, 43 * 100), (61 * 100, 52 * 100), (57 * 100, 48 * 100), (56 * 100, 37 * 100), (55 * 100, 54 * 100), (15 * 100, 47 * 100), (14 * 100, 37 * 100), (11 * 100, 31 * 100), (16 * 100, 22 * 100), (4 * 100, 18 * 100), (28 * 100, 18 * 100), (26 * 100, 52 * 100), (26 * 100, 35 * 100), (31 * 100, 67 * 100), (15 * 100, 19 * 100), (22 * 100, 22 * 100), (18 * 100, 24 * 100), (26 * 100, 27 * 100), (25 * 100, 24 * 100), (22 * 100, 27 * 100), (25 * 100, 21 * 100), (19 * 100, 21 * 100), (20 * 100, 26 * 100), (18 * 100, 18 * 100)]
FALLBACK_WEIGHTS = [0, 10, 7, 13, 19, 26, 3, 5, 9, 16, 16, 12, 19, 23, 20, 8, 19, 2, 12, 17, 9, 11, 18, 29, 3, 6, 17, 16, 16, 9, 21, 27, 23, 11, 14, 8, 5, 8, 16, 31, 9, 5, 5, 7, 18, 16, 1, 27, 36, 30, 13, 10, 9, 14, 18, 2, 6, 7, 18, 28, 3, 13, 19, 10, 9, 20, 25, 25, 36, 6, 5, 15, 25, 9, 8, 18, 13, 14, 3, 23, 6, 26, 16, 11, 7, 41, 35, 26, 9, 15, 3, 1, 2, 22, 27, 20, 11, 12, 10, 9, 17]
FALLBACK_TW = [(0, 2300), (0, 1710), (0, 600), (0, 1260), (0, 1590), (0, 440), (0, 1090), (0, 910), (0, 1050), (0, 1070), (0, 1340), (0, 770), (0, 730), (0, 1690), (0, 420), (0, 710), (0, 850), (0, 1670), (0, 970), (0, 860), (0, 1360), (0, 720), (0, 1070), (0, 780), (0, 1630), (0, 1820), (0, 1420), (0, 470), (0, 490), (0, 730), (0, 810), (0, 600), (0, 1510), (0, 470), (0, 1270), (0, 1530), (0, 510), (0, 1440), (0, 930), (0, 540), (0, 950), (0, 1070), (0, 410), (0, 1420), (0, 790), (0, 420), (0, 1270), (0, 610), (0, 1750), (0, 1180), (0, 1340), (0, 980), (0, 620), (0, 1050), (0, 1500), (0, 1460), (0, 1400), (0, 1110), (0, 2100), (0, 280), (0, 1720), (0, 860), (0, 680), (0, 440), (0, 830), (0, 610), (0, 1370), (0, 930), (0, 1520), (0, 600), (0, 1920), (0, 870), (0, 450), (0, 880), (0, 1590), (0, 790), (0, 830), (0, 1890), (0, 1060), (0, 1020), (0, 1920), (0, 1040), (0, 650), (0, 540), (0, 1110), (0, 1010), (0, 1040), (0, 1030), (0, 840), (0, 1860), (0, 1050), (0, 1700), (0, 280), (0, 1980), (0, 1100), (0, 490), (0, 1450), (0, 1430), (0, 680), (0, 930), (0, 1950)]

def parse_solomon_txt(file_path, n_customers):
    rows = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            nums = re.findall('-?\\d+', line)
            if len(nums) >= 7:
                rows.append(list(map(int, nums[:7])))
    rows = sorted(rows, key=lambda r: r[0])
    if len(rows) < n_customers + 1:
        raise ValueError('Instance {} has only {} customers, cannot take N={}.'.format(file_path, len(rows) - 1, n_customers))
    depot = rows[0]
    customers = rows[1:n_customers + 1]
    warehouse_xy = (depot[1] * SCALE, depot[2] * SCALE)
    positions = [(r[1] * SCALE, r[2] * SCALE) for r in customers]
    weights = [0] + [r[3] for r in customers]
    time_windows = [(0.0, float(depot[5] * 10))]
    time_windows.extend(((0.0, float(r[5] * 10)) for r in customers))
    return (warehouse_xy, positions, weights, time_windows)

def load_instance(instance_file, n_customers):
    path = SOLOMON_DIR / instance_file
    if path.exists():
        return parse_solomon_txt(path, n_customers)
    if instance_file.lower().startswith('r101'):
        return (FALLBACK_WAREHOUSE, FALLBACK_POSITIONS[:n_customers], FALLBACK_WEIGHTS[:n_customers + 1], FALLBACK_TW[:n_customers + 1])
    raise FileNotFoundError('{} not found, and no fallback is available for this instance.'.format(path))

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def attenuated_LA_at_point(src_pos, recv_pos):
    d = max(euclidean(src_pos, recv_pos), 0.1)
    return DRONE_LW - 20.0 * math.log10(d) - ALPHA * d

def build_facility_positions():
    rng = random.Random(SEED)
    return [(rng.uniform(0, 100 * SCALE), rng.uniform(0, 100 * SCALE)) for _ in range(100)]

def fleet_caps(n_customers):
    num_trucks = min(15, max(1, int(math.ceil(n_customers / 10.0))))
    num_direct_drones = 5
    return (num_trucks, num_direct_drones)

def calc_gap(lb, ub):
    if lb is None or ub is None:
        return None
    if abs(ub) < 1e-09:
        return 0.0 if abs(lb) < 1e-09 else None
    return max(0.0, (ub - lb) / abs(ub) * 100.0)

def recovery_candidates(i, j, nodes):
    if i == 0:
        return [p for p in nodes if p != j]
    return [p for p in nodes if p != i and p != j]

def run_gurobi_three_stage(instance_file, n_customers, stages='all', time_limit=TIME_LIMIT, output_flag=0):
    pre = ensure_precomputed(instance_file, n_customers, solomon_dir=SOLOMON_DIR, output_dir=BASE_DIR / 'precomputed_inputs', facility_seed=SEED, rebuild=False)
    warehouse_position = tuple(pre['warehouse_position'])
    customer_positions = [tuple(x) for x in pre['customer_positions']]
    weights = list(pre['weights'])
    time_windows = [tuple(x) for x in pre['time_windows']]
    num_trucks, num_direct_drones = fleet_caps(n_customers)
    V = range(num_trucks)
    K = range(num_direct_drones)
    Nodes = range(n_customers + 1)
    Customers = range(1, n_customers + 1)
    md = np.asarray(pre['manhattan_matrix'], dtype=float)
    ed = np.asarray(pre['euclidean_matrix'], dtype=float)
    theta_tr = np.asarray(pre['theta_tr'], dtype=float)
    theta_d = np.asarray(pre['theta_d'], dtype=float)
    E = [tw[0] for tw in time_windows]
    L = [tw[1] for tw in time_windows]
    car_ok = {int(k): int(v) for k, v in pre['car_ok'].items()}
    direct_ok = {int(k): int(v) for k, v in pre['direct_ok'].items()}
    direct_range_ok = {int(k): int(v) for k, v in pre['direct_range_ok'].items()}
    car_range_ok = np.asarray(pre['car_range_ok_matrix'], dtype=bool)
    recovery_candidates_map = pre.get('recovery_candidates', {})

    def get_Rij(i, j):
        return list(recovery_candidates_map.get((int(i), int(j)), recovery_candidates(i, j, Nodes)))
    model = gp.Model('three_stage_truck_drone_aligned')
    model.Params.TimeLimit = float(time_limit)
    model.Params.OutputFlag = int(output_flag)
    model.Params.MIPFocus = 1
    model.Params.Presolve = 2
    M = BIG_M
    x_tr = model.addVars(V, Nodes, Nodes, vtype=GRB.BINARY, name='x_tr')
    t_tr = model.addVars(V, Nodes, lb=0.0, vtype=GRB.CONTINUOUS, name='t_tr')
    t_ret = model.addVars(V, lb=0.0, vtype=GRB.CONTINUOUS, name='t_ret')
    use_tr = model.addVars(V, vtype=GRB.BINARY, name='use_tr')
    yd = model.addVars(K, Customers, vtype=GRB.BINARY, name='yd')
    t_depD = model.addVars(K, Customers, lb=0.0, vtype=GRB.CONTINUOUS, name='t_depD')
    t_arrD = model.addVars(K, Customers, lb=0.0, vtype=GRB.CONTINUOUS, name='t_arrD')
    t_retD = model.addVars(K, Customers, lb=0.0, vtype=GRB.CONTINUOUS, name='t_retD')
    use_dd = model.addVars(K, vtype=GRB.BINARY, name='use_dd')
    order_dd = model.addVars(K, Customers, Customers, vtype=GRB.BINARY, name='order_dd')
    z_cd = model.addVars(V, Nodes, Customers, vtype=GRB.BINARY, name='z_cd')
    w_cd = model.addVars(V, Nodes, Customers, Nodes, vtype=GRB.BINARY, name='w_cd')
    t_depCD = model.addVars(V, Nodes, Customers, lb=0.0, vtype=GRB.CONTINUOUS, name='t_depCD')
    t_cd_arr = model.addVars(V, Nodes, Customers, lb=0.0, vtype=GRB.CONTINUOUS, name='t_cd_arr')
    t_cd_ret = model.addVars(V, Nodes, Customers, Nodes, lb=0.0, vtype=GRB.CONTINUOUS, name='t_cd_ret')
    wait_cd = model.addVars(V, Nodes, Customers, Nodes, lb=0.0, vtype=GRB.CONTINUOUS, name='wait_cd')
    use_cd = model.addVars(V, vtype=GRB.BINARY, name='use_cd')
    tasks = [(i, j) for i in Nodes for j in Customers if i != j]
    order_cd_keys = [(v, i, j, ip, jp) for v in V for i, j in tasks for ip, jp in tasks if (i, j) != (ip, jp)]
    order_cd = model.addVars(order_cd_keys, vtype=GRB.BINARY, name='order_cd')
    T = model.addVars(Customers, lb=0.0, vtype=GRB.CONTINUOUS, name='T')
    P = model.addVars(Customers, lb=0.0, vtype=GRB.CONTINUOUS, name='P')
    Tmax = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='Tmax')
    for v in V:
        for i in Nodes:
            model.addConstr(x_tr[v, i, i] == 0, name='NoTruckSelf_{}_{}'.format(v, i))
        depot_depart = gp.quicksum((x_tr[v, 0, j] for j in Customers))
        depot_return = gp.quicksum((x_tr[v, i, 0] for i in Customers))
        model.addConstr(depot_depart <= 1, name='MaxDispatch_{}'.format(v))
        model.addConstr(depot_depart == depot_return, name='DepotBalance_{}'.format(v))
        model.addConstr(use_tr[v] == depot_depart, name='UseTruck_{}'.format(v))
        for h in Customers:
            model.addConstr(gp.quicksum((x_tr[v, i, h] for i in Nodes if i != h)) == gp.quicksum((x_tr[v, h, j] for j in Nodes if j != h)), name='TruckFlow_{}_{}'.format(v, h))
        model.addConstr(t_tr[v, 0] == 0.0, name='TruckStartTime_{}'.format(v))
        for j in Customers:
            model.addConstr(t_tr[v, j] >= theta_tr[0][j] - M * (1 - x_tr[v, 0, j]), name='TruckTimeFromDepotLB_{}_{}'.format(v, j))
            for i in Customers:
                if i == j:
                    continue
                model.addConstr(t_tr[v, j] >= t_tr[v, i] + SERVICE_TIME + theta_tr[i][j] - M * (1 - x_tr[v, i, j]), name='TruckTimeLB_{}_{}_{}'.format(v, i, j))
        for i in Customers:
            model.addConstr(t_ret[v] >= t_tr[v, i] + SERVICE_TIME + theta_tr[i][0] - M * (1 - x_tr[v, i, 0]), name='TruckReturnLB_{}_{}'.format(v, i))
            model.addConstr(t_ret[v] <= t_tr[v, i] + SERVICE_TIME + theta_tr[i][0] + M * (1 - x_tr[v, i, 0]), name='TruckReturnUB_{}_{}'.format(v, i))
    for j in Customers:
        truck_service = gp.quicksum((x_tr[v, i, j] for v in V for i in Nodes if i != j))
        direct_service = gp.quicksum((yd[k, j] for k in K))
        car_service = gp.quicksum((z_cd[v, i, j] for v in V for i in Nodes if i != j))
        model.addConstr(truck_service + direct_service + car_service == 1, name='ServeOnce_{}'.format(j))
        if direct_ok[j] == 0:
            model.addConstr(direct_service == 0, name='NoDirect_{}'.format(j))
        if car_ok[j] == 0:
            model.addConstr(car_service == 0, name='NoCarDrone_{}'.format(j))
    for k in K:
        total_direct_tasks = gp.quicksum((yd[k, j] for j in Customers))
        model.addConstr(use_dd[k] <= total_direct_tasks, name='UseDD_UB_{}'.format(k))
        for j in Customers:
            model.addConstr(use_dd[k] >= yd[k, j], name='UseDD_LB_{}_{}'.format(k, j))
            model.addConstr(yd[k, j] <= direct_range_ok[j], name='DirectRange_{}_{}'.format(k, j))
            model.addConstr(t_depD[k, j] <= M * yd[k, j], name='DDDepCap_{}_{}'.format(k, j))
            model.addConstr(t_arrD[k, j] >= t_depD[k, j] + theta_d[0][j] - M * (1 - yd[k, j]), name='DDArrLB_{}_{}'.format(k, j))
            model.addConstr(t_arrD[k, j] <= t_depD[k, j] + theta_d[0][j] + M * (1 - yd[k, j]), name='DDArrUB_{}_{}'.format(k, j))
            model.addConstr(t_retD[k, j] >= t_arrD[k, j] + theta_d[j][0] - M * (1 - yd[k, j]), name='DDRetLB_{}_{}'.format(k, j))
            model.addConstr(t_retD[k, j] <= t_arrD[k, j] + theta_d[j][0] + M * (1 - yd[k, j]), name='DDRetUB_{}_{}'.format(k, j))
        for i in Customers:
            model.addConstr(order_dd[k, i, i] == 0, name='DDOrderDiag_{}_{}'.format(k, i))
            for j in Customers:
                if i == j:
                    continue
                model.addConstr(order_dd[k, i, j] <= yd[k, i], name='DDOrderLink1_{}_{}_{}'.format(k, i, j))
                model.addConstr(order_dd[k, i, j] <= yd[k, j], name='DDOrderLink2_{}_{}_{}'.format(k, i, j))
                model.addConstr(t_depD[k, j] >= t_retD[k, i] - M * (1 - order_dd[k, i, j]), name='DDNoOverlap_{}_{}_{}'.format(k, i, j))
        for i in Customers:
            for j in Customers:
                if i >= j:
                    continue
                model.addConstr(order_dd[k, i, j] + order_dd[k, j, i] >= yd[k, i] + yd[k, j] - 1, name='DDOrderChooseLB_{}_{}_{}'.format(k, i, j))
                model.addConstr(order_dd[k, i, j] + order_dd[k, j, i] <= 1, name='DDOrderChooseUB_{}_{}_{}'.format(k, i, j))
    for v in V:
        total_car_tasks = gp.quicksum((z_cd[v, i, j] for i in Nodes for j in Customers if i != j))
        model.addConstr(use_cd[v] <= total_car_tasks, name='UseCD_UB_{}'.format(v))
        model.addConstr(use_cd[v] <= use_tr[v], name='UseCDNeedTruck_{}'.format(v))
        for i in Nodes:
            model.addConstr(gp.quicksum((z_cd[v, i, j] for j in Customers if i != j)) <= 1, name='OneLaunchPerNode_{}_{}'.format(v, i))
            for j in Customers:
                if i == j:
                    continue
                Rij_list = get_Rij(i, j)
                invalid_recoveries = [p for p in Nodes if p not in Rij_list]
                for p in invalid_recoveries:
                    model.addConstr(w_cd[v, i, j, p] == 0, name='InvalidRecovery_{}_{}_{}_{}'.format(v, i, j, p))
                model.addConstr(use_cd[v] >= z_cd[v, i, j], name='UseCD_LB_{}_{}_{}'.format(v, i, j))
                if i == 0:
                    model.addConstr(z_cd[v, i, j] <= use_tr[v], name='LaunchDepotNeedTruck_{}_{}'.format(v, j))
                else:
                    model.addConstr(z_cd[v, i, j] <= gp.quicksum((x_tr[v, h, i] for h in Nodes if h != i)), name='LaunchVisited_{}_{}_{}'.format(v, i, j))
                model.addConstr(gp.quicksum((w_cd[v, i, j, p] for p in Rij_list)) == z_cd[v, i, j], name='ChooseRecovery_{}_{}_{}'.format(v, i, j))
                model.addConstr(t_depCD[v, i, j] <= M * z_cd[v, i, j], name='CDDepCap_{}_{}_{}'.format(v, i, j))
                if i == 0:
                    model.addConstr(t_depCD[v, i, j] >= 0.0, name='CDDepDepotLB_{}_{}'.format(v, j))
                else:
                    model.addConstr(t_depCD[v, i, j] >= t_tr[v, i] - M * (1 - z_cd[v, i, j]), name='CDDepAfterTruck_{}_{}_{}'.format(v, i, j))
                model.addConstr(t_cd_arr[v, i, j] >= t_depCD[v, i, j] + theta_d[i][j] - M * (1 - z_cd[v, i, j]), name='CDArrLB_{}_{}_{}'.format(v, i, j))
                model.addConstr(t_cd_arr[v, i, j] <= t_depCD[v, i, j] + theta_d[i][j] + M * (1 - z_cd[v, i, j]), name='CDArrUB_{}_{}_{}'.format(v, i, j))
                for p in Rij_list:
                    model.addConstr(w_cd[v, i, j, p] <= int(car_range_ok[i, j, p]), name='CDRange_{}_{}_{}_{}'.format(v, i, j, p))
                    if p == 0:
                        model.addConstr(w_cd[v, i, j, p] <= use_tr[v], name='RecoverDepotNeedTruck_{}_{}_{}'.format(v, i, j))
                    else:
                        model.addConstr(w_cd[v, i, j, p] <= gp.quicksum((x_tr[v, h, p] for h in Nodes if h != p)), name='RecoverVisited_{}_{}_{}_{}'.format(v, i, j, p))
                    model.addConstr(t_cd_ret[v, i, j, p] >= t_cd_arr[v, i, j] + theta_d[j][p] - M * (1 - w_cd[v, i, j, p]), name='CDRetLB_{}_{}_{}_{}'.format(v, i, j, p))
                    model.addConstr(t_cd_ret[v, i, j, p] <= t_cd_arr[v, i, j] + theta_d[j][p] + M * (1 - w_cd[v, i, j, p]), name='CDRetUB_{}_{}_{}_{}'.format(v, i, j, p))
                    if p == 0:
                        model.addConstr(wait_cd[v, i, j, p] == 0.0, name='CDWaitDepotZero_{}_{}_{}'.format(v, i, j))
                    else:
                        model.addConstr(t_cd_ret[v, i, j, p] <= t_tr[v, p] + M * (1 - w_cd[v, i, j, p]), name='CDSyncNonDepot_{}_{}_{}_{}'.format(v, i, j, p))
                        model.addConstr(wait_cd[v, i, j, p] >= t_tr[v, p] - t_cd_ret[v, i, j, p] - M * (1 - w_cd[v, i, j, p]), name='CDWaitLB_{}_{}_{}_{}'.format(v, i, j, p))
                        model.addConstr(wait_cd[v, i, j, p] <= t_tr[v, p] - t_cd_ret[v, i, j, p] + M * (1 - w_cd[v, i, j, p]), name='CDWaitUB_{}_{}_{}_{}'.format(v, i, j, p))
                        model.addConstr(wait_cd[v, i, j, p] <= M * w_cd[v, i, j, p], name='CDWaitCap_{}_{}_{}_{}'.format(v, i, j, p))
        for idx_a in range(len(tasks)):
            i, j = tasks[idx_a]
            for idx_b in range(idx_a + 1, len(tasks)):
                ip, jp = tasks[idx_b]
                model.addConstr(order_cd[v, i, j, ip, jp] <= z_cd[v, i, j], name='CDOrderLink1_{}_{}_{}_{}_{}'.format(v, i, j, ip, jp))
                model.addConstr(order_cd[v, i, j, ip, jp] <= z_cd[v, ip, jp], name='CDOrderLink2_{}_{}_{}_{}_{}'.format(v, i, j, ip, jp))
                model.addConstr(order_cd[v, ip, jp, i, j] <= z_cd[v, ip, jp], name='CDOrderLink1Rev_{}_{}_{}_{}_{}'.format(v, ip, jp, i, j))
                model.addConstr(order_cd[v, ip, jp, i, j] <= z_cd[v, i, j], name='CDOrderLink2Rev_{}_{}_{}_{}_{}'.format(v, ip, jp, i, j))
                model.addConstr(order_cd[v, i, j, ip, jp] + order_cd[v, ip, jp, i, j] >= z_cd[v, i, j] + z_cd[v, ip, jp] - 1, name='CDChooseOrderLB_{}_{}_{}'.format(v, idx_a, idx_b))
                model.addConstr(order_cd[v, i, j, ip, jp] + order_cd[v, ip, jp, i, j] <= 1, name='CDChooseOrderUB_{}_{}_{}'.format(v, idx_a, idx_b))
                for p in get_Rij(i, j):
                    model.addConstr(t_depCD[v, ip, jp] >= t_cd_ret[v, i, j, p] + wait_cd[v, i, j, p] - M * (1 - order_cd[v, i, j, ip, jp]) - M * (1 - w_cd[v, i, j, p]), name='CDNoOverlap_{}_{}_{}_{}_{}_{}'.format(v, i, j, p, ip, jp))
                for p in get_Rij(ip, jp):
                    model.addConstr(t_depCD[v, i, j] >= t_cd_ret[v, ip, jp, p] + wait_cd[v, ip, jp, p] - M * (1 - order_cd[v, ip, jp, i, j]) - M * (1 - w_cd[v, ip, jp, p]), name='CDNoOverlapRev_{}_{}_{}_{}_{}_{}'.format(v, ip, jp, p, i, j))
    for j in Customers:
        for v in V:
            truck_in_j = gp.quicksum((x_tr[v, i, j] for i in Nodes if i != j))
            model.addConstr(T[j] >= t_tr[v, j] - M * (1 - truck_in_j), name='TTruckLB_{}_{}'.format(j, v))
            model.addConstr(T[j] <= t_tr[v, j] + M * (1 - truck_in_j), name='TTruckUB_{}_{}'.format(j, v))
            for i in Nodes:
                if i == j:
                    continue
                model.addConstr(T[j] >= t_cd_arr[v, i, j] - M * (1 - z_cd[v, i, j]), name='TCDLB_{}_{}_{}'.format(v, i, j))
                model.addConstr(T[j] <= t_cd_arr[v, i, j] + M * (1 - z_cd[v, i, j]), name='TCDUB_{}_{}_{}'.format(v, i, j))
        for k in K:
            model.addConstr(T[j] >= t_arrD[k, j] - M * (1 - yd[k, j]), name='TDDLB_{}_{}'.format(k, j))
            model.addConstr(T[j] <= t_arrD[k, j] + M * (1 - yd[k, j]), name='TDDUB_{}_{}'.format(k, j))
        model.addConstr(P[j] >= T[j] - L[j], name='PenaltyLB_{}'.format(j))
        model.addConstr(P[j] >= 0.0, name='PenaltyNN_{}'.format(j))
        model.addConstr(Tmax >= T[j], name='TmaxCust_{}'.format(j))
    for v in V:
        model.addConstr(Tmax >= t_ret[v], name='TmaxTruckReturn_{}'.format(v))
    variable_cost_expr = gp.quicksum((TRUCK_UNIT_COST * md[i][j] * x_tr[v, i, j] for v in V for i in Nodes for j in Nodes if i != j)) + gp.quicksum((DRONE_UNIT_COST * (2.0 * ed[0][j]) * yd[k, j] for k in K for j in Customers)) + gp.quicksum((CAR_DRONE_UNIT_COST * (ed[i][j] + ed[j][p]) * w_cd[v, i, j, p] for v in V for i in Nodes for j in Customers if i != j for p in get_Rij(i, j))) + gp.quicksum((WAIT_COST_PER_SEC * wait_cd[v, i, j, p] for v in V for i in Nodes for j in Customers if i != j for p in get_Rij(i, j) if p != 0))
    fixed_cost_expr = FIXED_TRUCK_COST * gp.quicksum((use_tr[v] for v in V)) + FIXED_DIRECT_DRONE_COST * gp.quicksum((use_dd[k] for k in K)) + FIXED_CAR_DRONE_COST * gp.quicksum((use_cd[v] for v in V))
    cost_expr = variable_cost_expr + fixed_cost_expr
    penalty_expr = gp.quicksum((P[j] for j in Customers))
    safe_inst = os.path.splitext(os.path.basename(instance_file))[0]
    solution_dir = 'Gurobi_Solutions_{}_N{}'.format(safe_inst, n_customers)
    os.makedirs(solution_dir, exist_ok=True)

    def _extract_truck_route(v):
        route = [0]
        curr = 0
        seen = set()
        while True:
            nxts = [j for j in Nodes if j != curr and x_tr[v, curr, j].X > 0.5]
            if not nxts:
                break
            nxt = nxts[0]
            route.append(nxt)
            if nxt == 0:
                break
            if nxt in seen:
                break
            seen.add(nxt)
            curr = nxt
        return route

    def dump_current_solution(tag):
        if model.SolCount <= 0:
            return None
        path = os.path.join(solution_dir, '{}_{}.txt'.format(tag, safe_inst))
        with open(path, 'w', encoding='utf-8') as f:
            f.write('=== {} | instance={} | N={} ===\n'.format(tag, instance_file, n_customers))
            f.write('ModelStatus: {}\n'.format(model.Status))
            try:
                f.write('ObjVal: {:.6f}\n'.format(model.ObjVal))
                f.write('CostExpr: {:.6f}\n'.format(cost_expr.getValue()))
                f.write('PenaltyExpr: {:.6f}\n'.format(penalty_expr.getValue()))
                f.write('Tmax: {:.6f}\n'.format(Tmax.X))
            except Exception:
                pass
            f.write('\n[Truck Routes]\n')
            has_truck = False
            for v in V:
                if use_tr[v].X > 0.5:
                    has_truck = True
                    route = _extract_truck_route(v)
                    f.write('Truck {}: route={} | return={:.3f}\n'.format(v, route, t_ret[v].X))
                    for node in route:
                        if node != 0:
                            f.write('  customer {}: t_tr={:.3f} | T={:.3f} | P={:.3f}\n'.format(node, t_tr[v, node].X, T[node].X, P[node].X))
            if not has_truck:
                f.write('None\n')
            f.write('\n[Direct Drone Tasks]\n')
            has_dd = False
            for k in K:
                tasks_k = [j for j in Customers if yd[k, j].X > 0.5]
                if tasks_k:
                    has_dd = True
                    tasks_k_sorted = sorted(tasks_k, key=lambda jj: t_depD[k, jj].X)
                    f.write('DirectDrone {}: tasks={}\n'.format(k, tasks_k_sorted))
                    for j in tasks_k_sorted:
                        f.write('  customer {}: t_dep={:.3f} | t_arr={:.3f} | t_ret={:.3f} | T={:.3f} | P={:.3f}\n'.format(j, t_depD[k, j].X, t_arrD[k, j].X, t_retD[k, j].X, T[j].X, P[j].X))
            if not has_dd:
                f.write('None\n')
            f.write('\n[Truck-mounted Drone Tasks]\n')
            has_cd = False
            for v in V:
                for i, j in tasks:
                    if z_cd[v, i, j].X > 0.5:
                        has_cd = True
                        rec_nodes = [p for p in get_Rij(i, j) if w_cd[v, i, j, p].X > 0.5]
                        rec = rec_nodes[0] if rec_nodes else None
                        ret_val = t_cd_ret[v, i, j, rec].X if rec is not None else float('nan')
                        wait_val = wait_cd[v, i, j, rec].X if rec is not None else float('nan')
                        f.write('Truck {}: launch={} -> customer={} -> recover={} | t_dep={:.3f} | t_arr={:.3f} | t_ret={:.3f} | wait={:.3f} | T={:.3f} | P={:.3f}\n'.format(v, i, j, rec, t_depCD[v, i, j].X, t_cd_arr[v, i, j].X, ret_val, wait_val, T[j].X, P[j].X))
            if not has_cd:
                f.write('None\n')
        return path
    results = {'Instance': instance_file, 'N': n_customers, 'TruckCap': num_trucks, 'DirectDroneCap': num_direct_drones, 'Cost_LB': None, 'Cost_UB': None, 'Cost_Gap(%)': None, 'Time_Cost(s)': 0.0, 'Status_Cost': '', 'Penalty_LB': None, 'Penalty_UB': None, 'Penalty_Gap(%)': None, 'Time_Penalty(s)': 0.0, 'Status_Penalty': '', 'Tmax_LB': None, 'Tmax_UB': None, 'Tmax_Gap(%)': None, 'Time_Tmax(s)': 0.0, 'Status_Tmax': '', 'Stage2_Ran': False, 'Time_Total(s)': 0.0}
    print('\n{}'.format('=' * 78))
    print('[{} | N={}] Gurobi three-stage: Cost -> Penalty -> Tmax | Penalty*=P*'.format(instance_file, n_customers))
    print('{}'.format('=' * 78))
    print('\n[Stage 0] Minimize Cost')
    model.setObjective(cost_expr, GRB.MINIMIZE)
    t0 = time.time()
    model.optimize()
    t1 = time.time()
    results['Time_Cost(s)'] = t1 - t0
    results['Status_Cost'] = str(model.Status)
    if model.Status == GRB.OPTIMAL:
        results['Cost_LB'] = model.ObjVal
        results['Cost_UB'] = model.ObjVal
        results['Cost_Gap(%)'] = 0.0
        print('Cost optimal = ${:.4f}'.format(model.ObjVal))
        dump_current_solution('Stage0_Cost_Optimal')
    elif model.Status == GRB.TIME_LIMIT:
        results['Cost_LB'] = model.ObjBound
        if model.SolCount > 0:
            results['Cost_UB'] = model.ObjVal
            results['Cost_Gap(%)'] = calc_gap(results['Cost_LB'], results['Cost_UB'])
            print('Cost time limit | LB=${:.4f}, UB=${:.4f}, Gap={:.2f}%'.format(results['Cost_LB'], results['Cost_UB'], results['Cost_Gap(%)'] if results['Cost_Gap(%)'] is not None else float('nan')))
            dump_current_solution('Stage0_Cost_BestIncumbent')
        else:
            print('Cost time limit | only LB=${:.4f}'.format(results['Cost_LB']))
    else:
        print('Cost stage failed. Status={}'.format(model.Status))
    if stages == 'cost_only':
        results['Time_Total(s)'] = results['Time_Cost(s)']
        return results
    print('\n[Stage 1] Minimize Penalty')
    model.setObjective(penalty_expr, GRB.MINIMIZE)
    t0 = time.time()
    model.optimize()
    t1 = time.time()
    results['Time_Penalty(s)'] = t1 - t0
    results['Status_Penalty'] = str(model.Status)
    penalty_star = None
    if model.Status == GRB.OPTIMAL:
        penalty_star = model.ObjVal
        results['Penalty_LB'] = model.ObjVal
        results['Penalty_UB'] = model.ObjVal
        results['Penalty_Gap(%)'] = 0.0
        print('Penalty optimal = {:.4f}'.format(model.ObjVal))
    elif model.Status == GRB.TIME_LIMIT:
        results['Penalty_LB'] = model.ObjBound
        if model.SolCount > 0:
            results['Penalty_UB'] = model.ObjVal
            results['Penalty_Gap(%)'] = calc_gap(results['Penalty_LB'], results['Penalty_UB'])
            print('Penalty time limit | LB={:.4f}, UB={:.4f}, Gap={:.2f}%'.format(results['Penalty_LB'], results['Penalty_UB'], results['Penalty_Gap(%)'] if results['Penalty_Gap(%)'] is not None else float('nan')))
        else:
            print('Penalty time limit | only LB={:.4f}'.format(results['Penalty_LB']))
    else:
        print('Penalty stage failed. Status={}'.format(model.Status))
    if penalty_star is not None:
        print('\n[Stage 2] Minimize Tmax with Penalty <= P*')
        results['Stage2_Ran'] = True
        model.addConstr(penalty_expr <= penalty_star + 1e-06, name='PenaltyOptCap')
        model.setObjective(Tmax, GRB.MINIMIZE)
        t0 = time.time()
        model.optimize()
        t1 = time.time()
        results['Time_Tmax(s)'] = t1 - t0
        results['Status_Tmax'] = str(model.Status)
        if model.Status == GRB.OPTIMAL:
            results['Tmax_LB'] = model.ObjVal
            results['Tmax_UB'] = model.ObjVal
            results['Tmax_Gap(%)'] = 0.0
            print('Tmax optimal = {:.4f}'.format(model.ObjVal))
            dump_current_solution('Stage2_Tmax_Optimal')
        elif model.Status == GRB.TIME_LIMIT:
            results['Tmax_LB'] = model.ObjBound
            if model.SolCount > 0:
                results['Tmax_UB'] = model.ObjVal
                results['Tmax_Gap(%)'] = calc_gap(results['Tmax_LB'], results['Tmax_UB'])
                print('Tmax time limit | LB={:.4f}, UB={:.4f}, Gap={:.2f}%'.format(results['Tmax_LB'], results['Tmax_UB'], results['Tmax_Gap(%)'] if results['Tmax_Gap(%)'] is not None else float('nan')))
                dump_current_solution('Stage2_Tmax_BestIncumbent')
            else:
                print('Tmax time limit | only LB={:.4f}'.format(results['Tmax_LB']))
        else:
            print('Tmax stage failed. Status={}'.format(model.Status))
    else:
        print('\nStage 1 did not prove an optimal penalty value; skip Stage 2.')
    results['Time_Total(s)'] = results['Time_Cost(s)'] + results['Time_Penalty(s)'] + results['Time_Tmax(s)']
    return results

def write_header(csv_filename):
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Instance', 'N', 'TruckCap', 'DirectDroneCap', 'Cost_LB($)', 'Cost_UB($)', 'Cost_Gap(%)', 'Time_Cost(s)', 'Status_Cost', 'Penalty_LB', 'Penalty_UB', 'Penalty_Gap(%)', 'Time_Penalty(s)', 'Status_Penalty', 'Tmax_LB', 'Tmax_UB', 'Tmax_Gap(%)', 'Time_Tmax(s)', 'Status_Tmax', 'Stage2_Ran', 'Time_Total(s)', 'TruckFixedCost($)', 'DirectDroneFixedCost($)', 'CarDroneFixedCost($)'])

def append_result(csv_filename, res):
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([res['Instance'], res['N'], res['TruckCap'], res['DirectDroneCap'], res['Cost_LB'], res['Cost_UB'], res['Cost_Gap(%)'], round(res['Time_Cost(s)'], 2), res['Status_Cost'], res['Penalty_LB'], res['Penalty_UB'], res['Penalty_Gap(%)'], round(res['Time_Penalty(s)'], 2), res['Status_Penalty'], res['Tmax_LB'], res['Tmax_UB'], res['Tmax_Gap(%)'], round(res['Time_Tmax(s)'], 2), res['Status_Tmax'], res['Stage2_Ran'], round(res['Time_Total(s)'], 2), round(FIXED_TRUCK_COST, 4), round(FIXED_DIRECT_DRONE_COST, 4), round(FIXED_CAR_DRONE_COST, 4)])
