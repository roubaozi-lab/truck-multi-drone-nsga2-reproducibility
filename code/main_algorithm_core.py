from __future__ import annotations
import random
import math
import time
import numpy as np
import pandas as pd
from deap.tools._hypervolume import hv
import heapq
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
ALGO_VERSION = 'v29_hpp_cost_front_boost_fast'
DEFAULT_CX_PB = 0.6
DEFAULT_BASE_MUT_PB = 0.15
DEFAULT_ELITE_RATIO = 0.15
SEED = 42
warehouse_position = (35 * 100, 35 * 100)
wait_time = 10.0
VEHICLE_SPEED = 10.0
vehicle_speed = 10.0
DRONE_SPEED = 20.0
VEHICLE_UNIT_COST = 1.5 / 7.3 / 1000.0
DRONE_UNIT_COST = 0.5 / 7.3 / 1000.0
CAR_DRONE_UNIT_COST = 0.5 / 7.3 / 1000.0
WAIT_COST_PER_SEC = CAR_DRONE_UNIT_COST * DRONE_SPEED
TRUCK_FIXED_COST = 100.0 / 7.3
DIRECT_DRONE_FIXED_COST = 20.0 / 7.3
CAR_DRONE_FIXED_COST = 20.0 / 7.3
ROUTE_EVAL_CACHE = {}
DRONE_SET_CACHE = {}
CAR_RANGE_OK_MATRIX = None
RECOVERY_CANDIDATES_MAP = None
drone_payload = 30
car_drone_payload = 30
scale = 100.0
FACILITY_NOISE_THRESHOLD_DB = 40.0
DRONE_MAX_RANGE = 10000
CAR_DRONE_MAX_RANGE = 10000
DRONE_LW = 86.6
ALPHA = 0.005
ORIGINAL_POSITIONS = [(41 * 100, 49 * 100), (35 * 100, 17 * 100), (55 * 100, 45 * 100), (55 * 100, 20 * 100), (15 * 100, 30 * 100), (25 * 100, 30 * 100), (20 * 100, 50 * 100), (10 * 100, 43 * 100), (55 * 100, 60 * 100), (30 * 100, 60 * 100), (20 * 100, 65 * 100), (50 * 100, 35 * 100), (30 * 100, 25 * 100), (15 * 100, 10 * 100), (30 * 100, 5 * 100), (10 * 100, 20 * 100), (5 * 100, 30 * 100), (20 * 100, 40 * 100), (15 * 100, 60 * 100), (45 * 100, 65 * 100), (45 * 100, 20 * 100), (45 * 100, 10 * 100), (55 * 100, 5 * 100), (65 * 100, 35 * 100), (65 * 100, 20 * 100), (45 * 100, 30 * 100), (35 * 100, 40 * 100), (41 * 100, 37 * 100), (64 * 100, 42 * 100), (40 * 100, 60 * 100), (31 * 100, 52 * 100), (35 * 100, 69 * 100), (53 * 100, 52 * 100), (65 * 100, 55 * 100), (63 * 100, 65 * 100), (2 * 100, 60 * 100), (20 * 100, 20 * 100), (5 * 100, 5 * 100), (60 * 100, 12 * 100), (40 * 100, 25 * 100), (42 * 100, 7 * 100), (24 * 100, 12 * 100), (23 * 100, 3 * 100), (11 * 100, 14 * 100), (6 * 100, 38 * 100), (2 * 100, 48 * 100), (8 * 100, 56 * 100), (13 * 100, 52 * 100), (6 * 100, 68 * 100), (47 * 100, 47 * 100), (49 * 100, 58 * 100), (27 * 100, 43 * 100), (37 * 100, 31 * 100), (57 * 100, 29 * 100), (63 * 100, 23 * 100), (53 * 100, 12 * 100), (32 * 100, 12 * 100), (36 * 100, 26 * 100), (21 * 100, 24 * 100), (17 * 100, 34 * 100), (12 * 100, 24 * 100), (24 * 100, 58 * 100), (27 * 100, 69 * 100), (15 * 100, 77 * 100), (62 * 100, 77 * 100), (49 * 100, 73 * 100), (67 * 100, 5 * 100), (56 * 100, 39 * 100), (37 * 100, 47 * 100), (37 * 100, 56 * 100), (57 * 100, 68 * 100), (47 * 100, 16 * 100), (44 * 100, 17 * 100), (46 * 100, 13 * 100), (49 * 100, 11 * 100), (49 * 100, 42 * 100), (53 * 100, 43 * 100), (61 * 100, 52 * 100), (57 * 100, 48 * 100), (56 * 100, 37 * 100), (55 * 100, 54 * 100), (15 * 100, 47 * 100), (14 * 100, 37 * 100), (11 * 100, 31 * 100), (16 * 100, 22 * 100), (4 * 100, 18 * 100), (28 * 100, 18 * 100), (26 * 100, 52 * 100), (26 * 100, 35 * 100), (31 * 100, 67 * 100), (15 * 100, 19 * 100), (22 * 100, 22 * 100), (18 * 100, 24 * 100), (26 * 100, 27 * 100), (25 * 100, 24 * 100), (22 * 100, 27 * 100), (25 * 100, 21 * 100), (19 * 100, 21 * 100), (20 * 100, 26 * 100), (18 * 100, 18 * 100)]
ORIGINAL_WEIGHTS = [0, 10, 7, 13, 19, 26, 3, 5, 9, 16, 16, 12, 19, 23, 20, 8, 19, 2, 12, 17, 9, 11, 18, 29, 3, 6, 17, 16, 16, 9, 21, 27, 23, 11, 14, 8, 5, 8, 16, 31, 9, 5, 5, 7, 18, 16, 1, 27, 36, 30, 13, 10, 9, 14, 18, 2, 6, 7, 18, 28, 3, 13, 19, 10, 9, 20, 25, 25, 36, 6, 5, 15, 25, 9, 8, 18, 13, 14, 3, 23, 6, 26, 16, 11, 7, 41, 35, 26, 9, 15, 3, 1, 2, 22, 27, 20, 11, 12, 10, 9, 17]
ORIGINAL_TW = [(0, 2300), (0, 1710), (0, 600), (0, 1260), (0, 1590), (0, 440), (0, 1090), (0, 910), (0, 1050), (0, 1070), (0, 1340), (0, 770), (0, 730), (0, 1690), (0, 420), (0, 710), (0, 850), (0, 1670), (0, 970), (0, 860), (0, 1360), (0, 720), (0, 1070), (0, 780), (0, 1630), (0, 1820), (0, 1420), (0, 470), (0, 490), (0, 730), (0, 810), (0, 600), (0, 1510), (0, 470), (0, 1270), (0, 1530), (0, 510), (0, 1440), (0, 930), (0, 540), (0, 950), (0, 1070), (0, 410), (0, 1420), (0, 790), (0, 420), (0, 1270), (0, 610), (0, 1750), (0, 1180), (0, 1340), (0, 980), (0, 620), (0, 1050), (0, 1500), (0, 1460), (0, 1400), (0, 1110), (0, 2100), (0, 280), (0, 1720), (0, 860), (0, 680), (0, 440), (0, 830), (0, 610), (0, 1370), (0, 930), (0, 1520), (0, 600), (0, 1920), (0, 870), (0, 450), (0, 880), (0, 1590), (0, 790), (0, 830), (0, 1890), (0, 1060), (0, 1020), (0, 1920), (0, 1040), (0, 650), (0, 540), (0, 1110), (0, 1010), (0, 1040), (0, 1030), (0, 840), (0, 1860), (0, 1050), (0, 1700), (0, 280), (0, 1980), (0, 1100), (0, 490), (0, 1450), (0, 1430), (0, 680), (0, 930), (0, 1950)]
facility_positions = [(35 * 100, 35 * 100), (45 * 100, 45 * 100)]

def calculate_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def calculate_distance_from_warehouse(pos):
    return calculate_distance(warehouse_position, pos)

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def attenuated_LA_at_point(src_pos, recv_pos):
    d = max(math.hypot(src_pos[0] - recv_pos[0], src_pos[1] - recv_pos[1]), 0.1)
    return DRONE_LW - 20 * math.log10(d) - ALPHA * d

def solve_vrptw_alns(veh_customers, customer_positions, time_windows, alpha=0.5, max_iter=10):
    local_nodes = [0] + list(veh_customers)
    local_pos = {node: idx for idx, node in enumerate(local_nodes)}
    dist_mat = precomputed_manhattan[np.ix_(local_nodes, local_nodes)]
    remaining = set(veh_customers)
    route = [0]
    T_cur = 0.0
    prev = 0
    while remaining:
        i_loc = local_pos[prev]
        delta, slack = ({}, {})
        for j in remaining:
            j_loc = local_pos[j]
            Delta = dist_mat[i_loc, j_loc] / VEHICLE_SPEED
            T_hat = T_cur + Delta
            _, L_j = time_windows[j]
            delta[j] = Delta
            slack[j] = max(0.0, L_j - T_hat)
        d_vals, s_vals = (delta.values(), slack.values())
        d_min, d_max = (min(d_vals), max(d_vals))
        s_min, s_max = (min(s_vals), max(s_vals))
        best_j, best_score = (None, float('inf'))
        for j in remaining:
            Dn = (delta[j] - d_min) / (d_max - d_min + 1e-06) if d_max > d_min else 0
            Sn = (s_max - slack[j]) / (s_max - s_min + 1e-06) if s_max > s_min else 0
            score = alpha * Dn + (1 - alpha) * Sn
            if score < best_score:
                best_score, best_j = (score, j)
        route.append(best_j)
        T_cur += delta[best_j]
        prev = best_j
        remaining.remove(best_j)
    route.append(0)
    local_route = [local_pos[g] for g in route]

    def evaluate_local_route(rt):
        t_curr, pen, dst = (0.0, 0.0, 0.0)
        max_node_pen = 0.0
        worst_n = None
        for i in range(len(rt) - 1):
            u, v = (rt[i], rt[i + 1])
            d = dist_mat[u][v]
            dst += d
            t_curr += d / VEHICLE_SPEED
            if v != 0:
                global_v = local_nodes[v]
                _, L_v = time_windows[global_v]
                if t_curr > L_v:
                    node_p = t_curr - L_v
                    pen += node_p
                    if node_p > max_node_pen:
                        max_node_pen = node_p
                        worst_n = v
                t_curr += wait_time
        return (pen, dst, worst_n)
    best_pen, best_dist, worst0 = evaluate_local_route(local_route)
    best_lr = local_route[:]
    if best_pen == 0:
        return ([local_nodes[idx] for idx in best_lr], dist_mat)
    current_lr, current_pen, current_dist, current_worst = (best_lr[:], best_pen, best_dist, worst0)
    temperature, cooling_rate = (100.0, 0.95)
    for iteration in range(max_iter):
        removed_node = current_worst if random.random() < 0.7 and current_worst is not None else None
        if removed_node is None and len(current_lr) > 3:
            removed_node = random.choice(current_lr[1:-1])
        if removed_node is None:
            break
        temp_lr = [n for n in current_lr if n != removed_node]
        best_repair, repair_pen, repair_dist, repair_worst = (None, float('inf'), float('inf'), None)
        for i in range(1, len(temp_lr)):
            test_lr = temp_lr[:i] + [removed_node] + temp_lr[i:]
            p, d, w = evaluate_local_route(test_lr)
            if p < repair_pen or (p == repair_pen and d < repair_dist):
                repair_pen, repair_dist, repair_worst = (p, d, w)
                best_repair = test_lr
        if repair_pen < current_pen or (repair_pen == current_pen and repair_dist < current_dist):
            current_lr, current_pen, current_dist, current_worst = (best_repair, repair_pen, repair_dist, repair_worst)
            if current_pen < best_pen or (current_pen == best_pen and current_dist < best_dist):
                best_lr, best_pen, best_dist = (current_lr, current_pen, current_dist)
        else:
            delta_E = (repair_pen - current_pen) * 100 + (repair_dist - current_dist)
            if delta_E > 0 and temperature > 0.01 and (random.random() < math.exp(-delta_E / temperature)):
                current_lr, current_pen, current_dist, current_worst = (best_repair, repair_pen, repair_dist, repair_worst)
        temperature *= cooling_rate
        if best_pen == 0:
            break
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_lr) - 2):
            for j in range(i + 1, len(best_lr) - 1):
                u, v = (best_lr[i - 1], best_lr[i])
                x, y = (best_lr[j], best_lr[j + 1])
                delta_d = dist_mat[u][x] + dist_mat[v][y] - dist_mat[u][v] - dist_mat[x][y]
                if delta_d < -1e-06:
                    test_opt = best_lr[:i] + best_lr[i:j + 1][::-1] + best_lr[j + 1:]
                    p, d, _ = evaluate_local_route(test_opt)
                    if p <= best_pen:
                        best_lr, best_pen, best_dist = (test_opt, p, d)
                        improved = True
                        break
            if improved:
                break
    return ([local_nodes[idx] for idx in best_lr], dist_mat)

def recovery_candidates(i, j, nodes):
    global RECOVERY_CANDIDATES_MAP
    if RECOVERY_CANDIDATES_MAP is not None:
        cand = RECOVERY_CANDIDATES_MAP.get((int(i), int(j)))
        if cand is not None:
            return list(cand)
    if i == 0:
        return [p for p in nodes if p != j]
    return [p for p in nodes if p != i and p != j]

def is_car_trip_range_feasible(launch_node, cust_idx, recovery_node):
    global CAR_RANGE_OK_MATRIX
    if CAR_RANGE_OK_MATRIX is not None:
        return bool(CAR_RANGE_OK_MATRIX[int(launch_node), int(cust_idx), int(recovery_node)])
    return PRECOMPUTED_EUCLIDEAN[launch_node, cust_idx] + PRECOMPUTED_EUCLIDEAN[cust_idx, recovery_node] <= CAR_DRONE_MAX_RANGE + 1e-09

def _build_car_range_ok_matrix(N):
    mat = np.zeros((N + 1, N + 1, N + 1), dtype=bool)
    nodes = range(N + 1)
    for launch_node in nodes:
        for cust_idx in range(1, N + 1):
            if launch_node == cust_idx:
                continue
            for recovery_node in recovery_candidates(launch_node, cust_idx, nodes):
                mat[launch_node, cust_idx, recovery_node] = PRECOMPUTED_EUCLIDEAN[launch_node, cust_idx] + PRECOMPUTED_EUCLIDEAN[cust_idx, recovery_node] <= CAR_DRONE_MAX_RANGE + 1e-09
    return mat

def has_any_car_range_feasible(cust_idx):
    global CAR_RANGE_OK_MATRIX
    if CAR_RANGE_OK_MATRIX is not None:
        return bool(np.any(CAR_RANGE_OK_MATRIX[:, int(cust_idx), :]))
    nodes = range(num_customers + 1)
    for launch_node in nodes:
        if launch_node == cust_idx:
            continue
        for recovery_node in recovery_candidates(launch_node, cust_idx, nodes):
            if is_car_trip_range_feasible(launch_node, cust_idx, recovery_node):
                return True
    return False

def evaluate_individual(individual):
    individual = _repair_orphan_car_assignments(individual)
    veh_c, car_c, drn_assign = ([], [], [])
    for i, gene in enumerate(individual, start=1):
        if gene[0] == 0:
            veh_c.append((gene[1], i))
        elif gene[0] == 1:
            drn_assign.append((gene[1], i))
        else:
            car_c.append((gene[1], gene[2], i))
    veh_cost, veh_time, veh_arrivals, vehicle_routes = (0.0, 0.0, {}, {})
    cust_by_vehicle = defaultdict(list)
    for vid, cust in veh_c:
        cust_by_vehicle[vid].append(cust)
    for vid, cl in cust_by_vehicle.items():
        route_key = tuple(sorted(cl))
        cached = ROUTE_EVAL_CACHE.get(route_key)
        if cached is None:
            rt, mat = solve_vrptw_alns(cl, customer_positions, time_windows, alpha=0.5, max_iter=10)
            c, t, arr = evaluate_vehicle(rt, mat, cl, wait_time)
            cached = (rt, c, t, arr)
            ROUTE_EVAL_CACHE[route_key] = cached
        rt, c, t, arr = cached
        vehicle_routes[vid] = rt
        veh_cost += c
        veh_time += t
        veh_arrivals.update(arr)
    carv_cost, carv_time, car_arrivals, drone_log = evaluate_car_drone(car_c, customer_positions, vehicle_routes, veh_arrivals, time_windows, wait_time=wait_time)
    drn_cost, drn_resource_finish, drn_arrivals = evaluate_drone(drn_assign, customer_positions, time_windows)
    all_arrivals = {**veh_arrivals, **car_arrivals, **drn_arrivals}
    vehicle_return_times = {}
    for vid, route in vehicle_routes.items():
        cust_route = [n for n in route if n != 0]
        if cust_route:
            last_cust = cust_route[-1]
            vehicle_return_times[vid] = veh_arrivals[last_cust] + wait_time + manhattan_distance(customer_positions[last_cust - 1], warehouse_position) / VEHICLE_SPEED
        else:
            vehicle_return_times[vid] = 0.0
    delivery_time = max(max(all_arrivals.values()) if all_arrivals else 0.0, max(vehicle_return_times.values()) if vehicle_return_times else 0.0)
    total_penalty = 0.0
    penalties = []
    for cust_idx in range(1, len(individual) + 1):
        T = all_arrivals.get(cust_idx, float('inf'))
        _, L_i = time_windows[cust_idx]
        if T > L_i:
            pen = T - L_i
            total_penalty += pen
            penalties.append((cust_idx, pen))
    sorted_penalties = sorted(penalties, key=lambda x: x[1], reverse=True)
    key = tuple(individual)
    WORST_NODE_DICT[key] = sorted_penalties[0][0] - 1 if sorted_penalties else None
    TOP_TARDY_NODES_DICT[key] = [cid - 1 for cid, _ in sorted_penalties[:6]]
    active_trucks = {vid for vid, _ in veh_c} | {vid for vid, _, _ in car_c}
    active_direct_drones = {did for did, _ in drn_assign}
    active_car_drones = {(vid, cdid) for vid, cdid, _ in car_c}
    fixed_cost = len(active_trucks) * TRUCK_FIXED_COST + len(active_direct_drones) * DIRECT_DRONE_FIXED_COST + len(active_car_drones) * CAR_DRONE_FIXED_COST
    total_cost = veh_cost + carv_cost + drn_cost + fixed_cost
    return (total_cost, delivery_time, total_penalty)

def custom_kmeans(data, k, n_init=5, max_iter=50):
    if k >= len(data):
        return [i % k for i in range(len(data))]
    best_inertia, best_labels = (float('inf'), None)
    for _ in range(n_init):
        centroids = random.sample(data, k)
        labels = [-1] * len(data)
        for _ in range(max_iter):
            changed = False
            clusters = [[] for _ in range(k)]
            for i, p in enumerate(data):
                c_idx = int(np.argmin([(p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 for c in centroids]))
                if labels[i] != c_idx:
                    labels[i] = c_idx
                    changed = True
                clusters[c_idx].append(p)
            if not changed:
                break
            for c_idx in range(k):
                if clusters[c_idx]:
                    centroids[c_idx] = (sum((p[0] for p in clusters[c_idx])) / len(clusters[c_idx]), sum((p[1] for p in clusters[c_idx])) / len(clusters[c_idx]))
        inertia = sum(((data[i][0] - centroids[labels[i]][0]) ** 2 + (data[i][1] - centroids[labels[i]][1]) ** 2 for i in range(len(data))))
        if inertia < best_inertia:
            best_inertia, best_labels = (inertia, labels[:])
    return best_labels

def dominates(a, b):
    return all((x <= y for x, y in zip(a, b))) and any((x < y for x, y in zip(a, b)))

def normalize_objs(pop_objs):
    arr = np.array(pop_objs, dtype=np.float32)
    mi, ma = (arr.min(axis=0), arr.max(axis=0))
    return ((arr - mi) / (ma - mi + 1e-09)).tolist()

def fast_nondominated_sort(objs):
    S, n, fronts = ([[] for _ in objs], [0] * len(objs), [[]])
    for p in range(len(objs)):
        for q in range(len(objs)):
            if dominates(objs[p], objs[q]):
                S[p].append(q)
            elif dominates(objs[q], objs[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    if not fronts[-1]:
        fronts.pop()
    return fronts

def crowding_distance(front, objs):
    dist = {i: 0.0 for i in front}
    for m in range(len(objs[0])):
        sf = sorted(front, key=lambda i: objs[i][m])
        f_min, f_max = (objs[sf[0]][m], objs[sf[-1]][m])
        dist[sf[0]] = dist[sf[-1]] = float('inf')
        if f_max == f_min:
            continue
        for idx in range(1, len(sf) - 1):
            dist[sf[idx]] += (objs[sf[idx + 1]][m] - objs[sf[idx - 1]][m]) / (f_max - f_min)
    return dist

def rebuild_penalty_maps(pop):
    WORST_NODE_DICT.clear()
    TOP_TARDY_NODES_DICT.clear()
    return [evaluate_individual(ind) for ind in pop]

def select_nsga2(pop, objs, k):
    fronts = fast_nondominated_sort(normalize_objs(objs))
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= k:
            selected.extend(front)
        else:
            dist = crowding_distance(front, objs)
            selected.extend(sorted(front, key=lambda i: -dist[i])[:k - len(selected)])
            break
    return ([pop[i] for i in selected], [objs[i] for i in selected])

def repair_individual(ind):
    for j, gene in enumerate(ind):
        if gene[0] not in customer_allowed[j]:
            new_type = random.choice(customer_allowed[j])
            if new_type == 0:
                ind[j] = (0, random.randrange(num_vehicles))
            elif new_type == 1:
                ind[j] = (1, random.randrange(num_drones))
            else:
                ind[j] = (2, random.randrange(num_vehicles), random.randrange(num_car_drones))
    return ind

def crossover(ind1, ind2):
    c1, c2 = (list(ind1), list(ind2))
    a, b = sorted(random.sample(range(len(ind1)), 2))
    c1[a:b], c2[a:b] = (c2[a:b], c1[a:b])
    return (repair_individual(c1), repair_individual(c2))

def evolve_one_generation(pop, objs, cx_pb, mut_pb, elite_memory, e_cx_pb, e_mut_pb, use_elite=True, stagnated=False):
    if use_elite:
        front = fast_nondominated_sort(normalize_objs(objs))[0]
        if front:
            dist = crowding_distance(front, objs)
            keep_n = max(1, int(math.ceil(len(front) * DEFAULT_ELITE_RATIO)))
            keep_idx = sorted(front, key=lambda i: -dist[i])[:keep_n]
            elite_pool = elite_memory + [repair_individual(list(pop[i])) for i in keep_idx]
            dedup = {}
            for ind in elite_pool:
                dedup[tuple(ind)] = ind
            elite_memory = list(dedup.values())[:100]
    offspring = []
    active_mut_pb = min(0.25, DEFAULT_BASE_MUT_PB * 3.0) if stagnated else DEFAULT_BASE_MUT_PB
    active_cx_pb = DEFAULT_CX_PB
    while len(offspring) < len(pop):
        i, j = random.sample(range(len(pop)), 2)
        p1 = pop[i] if dominates(objs[i], objs[j]) else pop[j]
        i, j = random.sample(range(len(pop)), 2)
        p2 = pop[i] if dominates(objs[i], objs[j]) else pop[j]
        c1, c2 = (list(p1), list(p2))
        if random.random() < active_cx_pb:
            if elite_memory and random.random() < e_cx_pb:
                elite = random.choice(elite_memory)
                a, b = sorted(random.sample(range(len(c1)), 2))
                c1[a:b], c2[a:b] = (elite[a:b], elite[a:b])
            else:
                c1, c2 = crossover(c1, c2)
        for c in [c1, c2]:
            if random.random() < active_mut_pb:
                if elite_memory and random.random() < e_mut_pb:
                    elite = random.choice(elite_memory)
                    pos = random.randrange(len(c))
                    c[pos] = elite[pos]
                else:
                    if stagnated:
                        strategies = ['TW', 'ECO', 'R']
                        weights = [30, 50, 20]
                    else:
                        strategies = ['TW', 'ECO', 'R']
                        weights = [45, 40, 15]
                    choice = random.choices(strategies, weights=weights, k=1)[0]
                    if choice == 'TW':
                        c, = time_window_aware_mutation(c)
                    elif choice == 'ECO':
                        c, = economic_structure_mutation(c)
                    else:
                        c, = random_mode_mutation(c)
            offspring.append(repair_individual(c))
    off_objs = [evaluate_individual(ind) for ind in offspring]
    return select_nsga2(pop + offspring, objs + off_objs, len(pop)) + (elite_memory,)

def _compute_csi_scores(candidate_idx):
    scores = []
    for i in candidate_idx:
        if 1 not in customer_allowed[i]:
            continue
        dist_depot = calculate_distance_from_warehouse(customer_positions[i])
        min_neighbor = min([calculate_distance(customer_positions[i], customer_positions[j]) for j in candidate_idx if i != j] + [1000000000.0])
        csi_score = min_neighbor * VEHICLE_UNIT_COST - dist_depot * 2 * DRONE_UNIT_COST
        scores.append((i, csi_score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def _cluster_centers(points, labels, k):
    centers = {}
    for c in range(k):
        cluster_pts = [points[j] for j in range(len(points)) if labels[j] == c]
        if cluster_pts:
            centers[c] = (sum((p[0] for p in cluster_pts)) / len(cluster_pts), sum((p[1] for p in cluster_pts)) / len(cluster_pts))
        else:
            centers[c] = points[random.randrange(len(points))]
    return centers

def _assign_cluster_modes(unassigned_idx, k, bridge_ratio=0.0):
    if not unassigned_idx:
        return {}
    unassigned_pos = [customer_positions[i] for i in unassigned_idx]
    if len(unassigned_idx) <= k:
        labels = list(range(len(unassigned_idx)))
        k_eff = len(unassigned_idx)
    else:
        labels = custom_kmeans(unassigned_pos, k=k)
        k_eff = k
    centers = _cluster_centers(unassigned_pos, labels, k_eff)
    assign = {}
    for c in range(k_eff):
        cluster_nodes = [unassigned_idx[j] for j in range(len(unassigned_idx)) if labels[j] == c]
        cluster_nodes.sort(key=lambda x: calculate_distance(customer_positions[x], centers[c]), reverse=True)
        bridge_candidates = [i for i in cluster_nodes if 2 in customer_allowed[i]]
        bridge_limit = int(round(len(cluster_nodes) * bridge_ratio))
        bridge_limit = min(len(bridge_candidates), bridge_limit)
        bridge_set = set(bridge_candidates[:bridge_limit])
        for global_i in cluster_nodes:
            if global_i in bridge_set:
                assign[global_i] = (2, c, random.randrange(num_car_drones))
            else:
                assign[global_i] = (0, c)
    return assign

def _mode_caps(n_customers):
    if n_customers <= 20:
        return (1, 1)
    if n_customers <= 40:
        return (2, 2)
    return (max(2, math.ceil(n_customers * 0.05)), max(2, math.ceil(n_customers * 0.08)))

def _build_truck_kmeans_band_ind(n_customers, gap_low=0, gap_high=2):
    ind = [None] * n_customers
    k_trucks = _sample_k_from_band(n_customers, gap_low, gap_high)
    assignments = _assign_cluster_modes(list(range(n_customers)), k_trucks, bridge_ratio=0.0)
    for idx, gene in assignments.items():
        ind[idx] = (0, gene[1])
    return repair_individual(ind)

def _build_truck_only_kmeans_ind(n_customers):
    ind = [None] * n_customers
    k_trucks = _sample_active_k(n_customers, spread=5)
    assignments = _assign_cluster_modes(list(range(n_customers)), k_trucks, bridge_ratio=0.0)
    for idx, gene in assignments.items():
        ind[idx] = (0, gene[1])
    return repair_individual(ind)

def _cluster_centers_from_ind(ind):
    truck_nodes = defaultdict(list)
    for i, gene in enumerate(ind):
        if gene[0] in (0, 2):
            truck_nodes[gene[1]].append(i)
    centers = {}
    for vid, nodes in truck_nodes.items():
        if nodes:
            centers[vid] = (sum((customer_positions[i][0] for i in nodes)) / len(nodes), sum((customer_positions[i][1] for i in nodes)) / len(nodes))
    return (truck_nodes, centers)

def _nearest_truck_vid(idx, centers):
    if not centers:
        return 0
    return min(centers.keys(), key=lambda vid: calculate_distance(customer_positions[idx], centers[vid]))

def promote_profit_guided_modes(ind, direct_ratio=0.08, car_ratio=0.08):
    ind = list(ind)
    n_customers = len(ind)
    max_direct, max_car = _mode_caps(n_customers)
    csi_scores = _compute_csi_scores(list(range(n_customers)))
    positive_direct = [i for i, sc in csi_scores if sc > 0 and ind[i][0] == 0 and (1 in customer_allowed[i])]
    direct_limit = min(max_direct, max(0, int(round(n_customers * direct_ratio))))
    for i in positive_direct[:direct_limit]:
        ind[i] = (1, sample_direct_id(ind))
    truck_nodes, centers = _cluster_centers_from_ind(ind)
    car_candidates = []
    for vid, nodes in truck_nodes.items():
        if not nodes:
            continue
        center = centers[vid]
        for i in nodes:
            if ind[i][0] == 0 and 2 in customer_allowed[i]:
                edge_score = calculate_distance(customer_positions[i], center)
                car_candidates.append((edge_score, i, vid))
    car_candidates.sort(reverse=True)
    car_limit = min(max_car, max(0, int(round(n_customers * car_ratio))))
    for _, i, vid in car_candidates[:car_limit]:
        if ind[i][0] == 0:
            ind[i] = (2, vid, 0)
    return repair_individual(ind)

def economic_screening_repair(ind):
    ind = list(ind)
    n_customers = len(ind)
    max_direct, max_car = _mode_caps(n_customers)
    csi_map = {i: sc for i, sc in _compute_csi_scores(list(range(n_customers)))}
    truck_nodes, centers = _cluster_centers_from_ind(ind)
    directs = [i for i, gene in enumerate(ind) if gene[0] == 1]
    directs_sorted = sorted(directs, key=lambda i: csi_map.get(i, -1e+18), reverse=True)
    keep_direct = set(directs_sorted[:max_direct])
    for i in directs:
        if i not in keep_direct or csi_map.get(i, -1e+18) <= 0:
            vid = _nearest_truck_vid(i, centers)
            ind[i] = (0, vid)
    truck_nodes, centers = _cluster_centers_from_ind(ind)
    cars = [i for i, gene in enumerate(ind) if gene[0] == 2]

    def car_score(i):
        vid = ind[i][1]
        center = centers.get(vid, customer_positions[i])
        return calculate_distance(customer_positions[i], center)
    cars_sorted = sorted(cars, key=car_score, reverse=True)
    keep_car = set(cars_sorted[:max_car])
    for i in cars:
        vid = ind[i][1]
        if i not in keep_car or len(truck_nodes.get(vid, [])) <= 1:
            ind[i] = (0, vid)
    return repair_individual(ind)

def air_service_counts(ind):
    d_cnt = sum((1 for g in ind if g[0] == 1))
    c_cnt = sum((1 for g in ind if g[0] == 2))
    return (d_cnt, c_cnt)

def _select_from_indices_nsga(indices, objs, k):
    if k <= 0 or not indices:
        return []
    if len(indices) <= k:
        return indices[:]
    sub_objs = [objs[i] for i in indices]
    fronts = fast_nondominated_sort(normalize_objs(sub_objs))
    chosen_local = []
    for front in fronts:
        front_global = [indices[i] for i in front]
        if len(chosen_local) + len(front_global) <= k:
            chosen_local.extend(front_global)
        else:
            dist = crowding_distance(front_global, objs)
            chosen_local.extend(sorted(front_global, key=lambda i: -dist[i])[:k - len(chosen_local)])
            break
    return chosen_local

def inject_service_archive(pop, objs, service_archive, replace_k=4):
    if not service_archive:
        return (pop, objs)
    worst_idx = sorted(range(len(objs)), key=lambda i: (objs[i][2], objs[i][1], objs[i][0]), reverse=True)[:min(replace_k, len(service_archive))]
    for wi, (ind, obj) in zip(worst_idx, service_archive[:len(worst_idx)]):
        pop[wi] = list(ind)
        objs[wi] = obj
    return (pop, objs)

def update_cost_archive(cost_archive, pop, objs, cap=12):
    candidates = cost_archive[:]
    candidates += [(list(pop[i]), objs[i]) for i in range(len(pop))]
    uniq = {}
    for ind, obj in candidates:
        uniq[tuple(ind)] = (list(ind), obj)
    merged = list(uniq.values())
    merged.sort(key=lambda x: (x[1][0], x[1][2], x[1][1]))
    return merged[:cap]

def update_high_penalty_streak(high_pen_streak, pop, objs, best_pen):
    threshold = max(200.0, 5.0 * max(best_pen, 1.0))
    present = set()
    for ind, obj in zip(pop, objs):
        key = tuple(ind)
        present.add(key)
        if obj[2] > threshold:
            high_pen_streak[key] = high_pen_streak.get(key, 0) + 1
        else:
            high_pen_streak[key] = 0
    for key in list(high_pen_streak.keys()):
        if key not in present:
            del high_pen_streak[key]
    return high_pen_streak

def run_kmeans_init_nsga2(s, N):
    return run_nsga2_pure(s, N, init_population_kmeans_std, True, n_gen=100, pop_size=100)

def run_hpp_init_nsga2(s, N):
    return run_nsga2_pure(s, N, init_population_hpp, True, n_gen=100, pop_size=100)

def set_environment_from_precomputed(pre):
    global customer_positions, weights, time_windows, num_customers
    global num_vehicles, num_drones, num_car_drones, customer_allowed
    global WORST_NODE_DICT, TOP_TARDY_NODES_DICT, facility_positions
    global ROUTE_EVAL_CACHE, DRONE_SET_CACHE
    global precomputed_manhattan, PRECOMPUTED_EUCLIDEAN
    global CAR_RANGE_OK_MATRIX, RECOVERY_CANDIDATES_MAP, warehouse_position
    warehouse_position = tuple(pre['warehouse_position'])
    customer_positions = [tuple(p) for p in pre['customer_positions']]
    weights = list(pre['weights'])
    time_windows = [tuple(tw) for tw in pre['time_windows']]
    facility_positions = [tuple(p) for p in pre.get('facility_positions', [])]
    num_customers = int(pre['N'])
    num_vehicles, num_drones = fleet_caps(num_customers)
    num_car_drones = int(pre.get('num_car_drones', 1))
    precomputed_manhattan = np.asarray(pre['manhattan_matrix'], dtype=float)
    PRECOMPUTED_EUCLIDEAN = np.asarray(pre['euclidean_matrix'], dtype=float)
    CAR_RANGE_OK_MATRIX = np.asarray(pre['car_range_ok_matrix'], dtype=bool)
    RECOVERY_CANDIDATES_MAP = pre.get('recovery_candidates', None)
    customer_allowed = [list(x) for x in pre['customer_allowed']]
    WORST_NODE_DICT = {}
    TOP_TARDY_NODES_DICT = {}
    ROUTE_EVAL_CACHE = {}
    DRONE_SET_CACHE = {}

def set_environment_for_N(N):
    global customer_positions, weights, time_windows, num_customers, num_vehicles, num_drones, num_car_drones, customer_allowed, WORST_NODE_DICT, TOP_TARDY_NODES_DICT, facility_positions, ROUTE_EVAL_CACHE, DRONE_SET_CACHE, CAR_RANGE_OK_MATRIX, RECOVERY_CANDIDATES_MAP
    customer_positions = ORIGINAL_POSITIONS[:N]
    weights = ORIGINAL_WEIGHTS[:N + 1]
    time_windows = ORIGINAL_TW[:N + 1]
    num_customers = N
    num_vehicles, num_drones = fleet_caps(N)
    num_car_drones = 1
    random.seed(SEED)
    facility_positions = [(random.uniform(0, 100 * scale), random.uniform(0, 100 * scale)) for _ in range(100)]
    precompute_data(customer_positions)
    RECOVERY_CANDIDATES_MAP = None
    CAR_RANGE_OK_MATRIX = _build_car_range_ok_matrix(N)
    WORST_NODE_DICT = {}
    TOP_TARDY_NODES_DICT = {}
    ROUTE_EVAL_CACHE = {}
    DRONE_SET_CACHE = {}
    customer_allowed = []
    for idx in range(1, N + 1):
        pos = customer_positions[idx - 1]
        w = weights[idx]
        allow = [0]
        noise_ok = all((attenuated_LA_at_point(pos, fpos) <= FACILITY_NOISE_THRESHOLD_DB for fpos in facility_positions))
        if w <= car_drone_payload and noise_ok:
            if has_any_car_range_feasible(idx):
                allow.append(2)
            if calculate_distance_from_warehouse(pos) * 2 <= DRONE_MAX_RANGE:
                allow.append(1)
        customer_allowed.append(allow)

def visualize_solution(individual, customer_positions, warehouse_position, time_windows, filename='VRP_Solution.png'):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from collections import defaultdict
    veh_c, car_c, drn_assign = ([], [], [])
    for i, gene in enumerate(individual, start=1):
        if gene[0] == 0:
            veh_c.append((gene[1], i))
        elif gene[0] == 1:
            drn_assign.append((gene[1], i))
        else:
            car_c.append((gene[1], gene[2], i))
    vehicle_routes = {}
    veh_arrivals = {}
    cust_by_vehicle = defaultdict(list)
    for vid, cust in veh_c:
        cust_by_vehicle[vid].append(cust)
    for vid, cl in cust_by_vehicle.items():
        rt, mat = solve_vrptw_alns(cl, customer_positions, time_windows)
        c, t, arr = evaluate_vehicle(rt, mat, cl, wait_time)
        vehicle_routes[vid] = rt
        veh_arrivals.update(arr)
    _, _, _, drone_log = evaluate_car_drone(car_c, customer_positions, vehicle_routes, veh_arrivals, time_windows, wait_time=wait_time)
    drn_c = [cust for _, cust in drn_assign]
    plt.figure(figsize=(14, 14))
    colors = list(mcolors.TABLEAU_COLORS.values()) * 3
    plt.scatter(warehouse_position[0], warehouse_position[1], c='red', marker='*', s=600, zorder=10, edgecolors='black', label='Depot')
    for cid in drn_c:
        pos = customer_positions[cid - 1]
        plt.plot([warehouse_position[0], pos[0]], [warehouse_position[1], pos[1]], color='#2ca02c', linestyle='--', alpha=0.3, linewidth=1.5)
        plt.scatter(pos[0], pos[1], c='#2ca02c', marker='^', s=120, edgecolors='black', zorder=5)
    if drn_c:
        plt.scatter([], [], c='#2ca02c', marker='^', s=120, edgecolors='black', label='Direct Drone')
    truck_legend, car_drone_legend = (False, False)
    for vid, route in vehicle_routes.items():
        if len(route) <= 2 and (not drone_log.get((vid, 0))):
            continue
        color = colors[vid % len(colors)]
        rx = [warehouse_position[0] if n == 0 else customer_positions[n - 1][0] for n in route]
        ry = [warehouse_position[1] if n == 0 else customer_positions[n - 1][1] for n in route]
        plt.plot(rx, ry, color=color, linewidth=2.5, linestyle='-', alpha=0.8)
        tx = [customer_positions[n - 1][0] for n in route if n != 0]
        ty = [customer_positions[n - 1][1] for n in route if n != 0]
        plt.scatter(tx, ty, c=color, marker='o', s=100, edgecolors='black', zorder=6)
        if not truck_legend:
            plt.scatter([], [], c='gray', marker='o', edgecolors='black', s=100, label='Truck')
            truck_legend = True
    for key, logs in drone_log.items():
        vid, cdid = key
        color = colors[vid % len(colors)]
        for log in logs:
            rel, ret, cust_idx, _, _, _, _ = log
            rel_pos = warehouse_position if rel == 0 else customer_positions[rel - 1]
            ret_pos = warehouse_position if ret == 0 else customer_positions[ret - 1]
            cust_pos = customer_positions[cust_idx - 1]
            plt.annotate('', xy=cust_pos, xytext=rel_pos, arrowprops=dict(arrowstyle='->', color=color, linestyle=':', lw=2.5, alpha=0.9))
            plt.annotate('', xy=ret_pos, xytext=cust_pos, arrowprops=dict(arrowstyle='->', color=color, linestyle=':', lw=2.5, alpha=0.9))
            plt.scatter(cust_pos[0], cust_pos[1], c=color, marker='s', s=120, edgecolors='black', zorder=7)
            if not car_drone_legend:
                plt.scatter([], [], c='gray', marker='s', edgecolors='black', s=120, label='Car-mounted Drone')
                car_drone_legend = True
    plt.title('Optimal Multi-Modal Routing Topology (HPP Algorithm)', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('X Coordinate (m)', fontsize=14)
    plt.ylabel('Y Coordinate (m)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.0, fontsize=12, frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close()
import os

def summarize_front_modes(pareto_front, pareto_objs=None):
    truck_services, direct_services, car_services = ([], [], [])
    active_trucks, active_directs, active_cars = ([], [], [])
    pure_truck_count = 0
    zero_penalty_count = 0
    zero_penalty_time_best = None
    zero_penalty_cost_best = None
    for idx, ind in enumerate(pareto_front):
        truck_services.append(sum((1 for g in ind if g[0] == 0)))
        direct_services.append(sum((1 for g in ind if g[0] == 1)))
        car_services.append(sum((1 for g in ind if g[0] == 2)))
        tset = {g[1] for g in ind if g[0] == 0} | {g[1] for g in ind if g[0] == 2}
        dset = {g[1] for g in ind if g[0] == 1}
        cset = {(g[1], g[2]) for g in ind if g[0] == 2}
        active_trucks.append(len(tset))
        active_directs.append(len(dset))
        active_cars.append(len(cset))
        if sum((1 for g in ind if g[0] in (1, 2))) == 0:
            pure_truck_count += 1
        if pareto_objs is not None:
            pen = pareto_objs[idx][2]
            if pen <= 1e-06:
                zero_penalty_count += 1
                if zero_penalty_time_best is None or pareto_objs[idx][1] < zero_penalty_time_best:
                    zero_penalty_time_best = pareto_objs[idx][1]
                if zero_penalty_cost_best is None or pareto_objs[idx][0] < zero_penalty_cost_best:
                    zero_penalty_cost_best = pareto_objs[idx][0]
    n = len(pareto_front) if pareto_front else 1
    return {'TruckService_Avg': round(float(np.mean(truck_services)), 6) if truck_services else 0.0, 'DirectService_Avg': round(float(np.mean(direct_services)), 6) if direct_services else 0.0, 'CarService_Avg': round(float(np.mean(car_services)), 6) if car_services else 0.0, 'AirService_Avg': round(float(np.mean(np.array(direct_services) + np.array(car_services))), 6) if truck_services else 0.0, 'ActiveTruck_Avg': round(float(np.mean(active_trucks)), 6) if active_trucks else 0.0, 'ActiveDirect_Avg': round(float(np.mean(active_directs)), 6) if active_directs else 0.0, 'ActiveCar_Avg': round(float(np.mean(active_cars)), 6) if active_cars else 0.0, 'PureTruck_Ratio': round(pure_truck_count / n, 6), 'ZeroPenalty_Count': zero_penalty_count, 'ZeroPenalty_Ratio': round(zero_penalty_count / n, 6), 'ZeroPenaltyTime_Best': round(float(zero_penalty_time_best), 6) if zero_penalty_time_best is not None else None, 'ZeroPenaltyCost_Best': round(float(zero_penalty_cost_best), 6) if zero_penalty_cost_best is not None else None}

def summarize_individual_modes(ind):
    truck_service = sum((1 for g in ind if g[0] == 0))
    direct_service = sum((1 for g in ind if g[0] == 1))
    car_service = sum((1 for g in ind if g[0] == 2))
    active_trucks = len({g[1] for g in ind if g[0] == 0} | {g[1] for g in ind if g[0] == 2})
    active_direct = len({g[1] for g in ind if g[0] == 1})
    active_car = len({(g[1], g[2]) for g in ind if g[0] == 2})
    return {'TruckService_Count': truck_service, 'DirectService_Count': direct_service, 'CarService_Count': car_service, 'AirService_Count': direct_service + car_service, 'ActiveTruck_Count': active_trucks, 'ActiveDirect_Count': active_direct, 'ActiveCar_Count': active_car}

def collect_key_chromosomes(pareto_front, pareto_objs, algo_name, N, run_id):
    rows = []
    if not pareto_front:
        return rows
    best_cost_idx = min(range(len(pareto_objs)), key=lambda i: pareto_objs[i][0])
    best_time_idx = min(range(len(pareto_objs)), key=lambda i: pareto_objs[i][1])
    best_pen_idx = min(range(len(pareto_objs)), key=lambda i: pareto_objs[i][2])
    key_cases = [('MinCost', best_cost_idx), ('MinTime', best_time_idx), ('MinPenalty', best_pen_idx)]
    for tag, idx in key_cases:
        ind = pareto_front[idx]
        obj = pareto_objs[idx]
        mode_info = summarize_individual_modes(ind)
        rows.append({'N': N, 'Run': run_id, 'Algorithm': algo_name, 'Version': ALGO_VERSION, 'KeyType': tag, 'Cost': round(obj[0], 6), 'Time': round(obj[1], 6), 'Penalty': round(obj[2], 6), 'TruckService_Count': mode_info['TruckService_Count'], 'DirectService_Count': mode_info['DirectService_Count'], 'CarService_Count': mode_info['CarService_Count'], 'Chromosome': str(ind)})
    return rows

def _decode_chromosome(chromosome):
    veh_c, car_c, drn_assign = ([], [], [])
    for i, gene in enumerate(chromosome, start=1):
        if gene[0] == 0:
            veh_c.append((gene[1], i))
        elif gene[0] == 1:
            drn_assign.append((gene[1], i))
        else:
            car_c.append((gene[1], gene[2], i))
    return (veh_c, car_c, drn_assign)

def _rebuild_solution_for_plot(chromosome):
    chromosome = _repair_orphan_car_assignments(chromosome)
    veh_c, car_c, drn_assign = _decode_chromosome(chromosome)
    veh_arrivals, vehicle_routes = ({}, {})
    cust_by_vehicle = defaultdict(list)
    for vid, cust in veh_c:
        cust_by_vehicle[vid].append(cust)
    for vid, cl in cust_by_vehicle.items():
        rt, mat = solve_vrptw_alns(cl, customer_positions, time_windows, alpha=0.5, max_iter=10)
        _, _, arr = evaluate_vehicle(rt, mat, cl, wait_time)
        vehicle_routes[vid] = rt
        veh_arrivals.update(arr)
    _, _, _, drone_log = evaluate_car_drone(car_c, customer_positions, vehicle_routes, veh_arrivals, time_windows, wait_time=wait_time)
    return (vehicle_routes, drone_log, drn_assign)

def export_key_chromosome_plots(key_df, folder_name):
    import ast
    hpp_df = key_df[key_df['Algorithm'] == 'HPP-init-NSGA2']
    for _, row in hpp_df.iterrows():
        set_environment_for_N(int(row['N']))
        chrom = ast.literal_eval(row['Chromosome'])
        name = f"{row['Algorithm']}_Run{row['Run']}_{row['KeyType']}.png".replace(' ', '_')
        save_path = os.path.join(folder_name, name)
        title = f"{row['Algorithm']} | Run {row['Run']} | {row['KeyType']} | C={row['Cost']:.2f}, T={row['Time']:.1f}, P={row['Penalty']:.1f}"
        plot_chromosome_solution(chrom, title, save_path)

def build_frontier_dataframe(pareto_front, pareto_objs, algo_name, N, run_id, front_tag, pen_cap=None):
    rows = []
    for idx, (ind, obj) in enumerate(zip(pareto_front, pareto_objs), start=1):
        mode_info = summarize_individual_modes(ind)
        rows.append({'N': N, 'Run': run_id, 'Algorithm': algo_name, 'Version': ALGO_VERSION, 'FrontType': front_tag, 'PenaltyCap': pen_cap, 'PointID': idx, 'Cost': round(obj[0], 6), 'Time': round(obj[1], 6), 'Penalty': round(obj[2], 6), 'TruckService_Count': mode_info['TruckService_Count'], 'DirectService_Count': mode_info['DirectService_Count'], 'CarService_Count': mode_info['CarService_Count'], 'AirService_Count': mode_info['AirService_Count'], 'ActiveTruck_Count': mode_info['ActiveTruck_Count'], 'ActiveDirect_Count': mode_info['ActiveDirect_Count'], 'ActiveCar_Count': mode_info['ActiveCar_Count'], 'Chromosome': str(ind)})
    return pd.DataFrame(rows)

def plot_frontier_3d(pareto_objs, title, save_path):
    if not pareto_objs:
        return
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    xs = [o[0] for o in pareto_objs]
    ys = [o[1] for o in pareto_objs]
    zs = [o[2] for o in pareto_objs]
    ax.scatter(xs, ys, zs, s=28, depthshade=True)
    ax.set_xlabel('Cost')
    ax.set_ylabel('Time')
    ax.set_zlabel('Penalty')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()

def plot_frontier_2d(pareto_objs, title, save_path):
    if not pareto_objs:
        return
    costs = [o[0] for o in pareto_objs]
    times = [o[1] for o in pareto_objs]
    pens = [o[2] for o in pareto_objs]
    plt.figure(figsize=(7.5, 6))
    sc = plt.scatter(costs, times, c=pens, s=34)
    plt.xlabel('Cost')
    plt.ylabel('Time')
    plt.title(title)
    cbar = plt.colorbar(sc)
    cbar.set_label('Penalty')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()

def export_frontier_bundle(pareto_front, pareto_objs, algo_name, N, run_id, front_tag, folder_name, pen_cap=None):
    df = build_frontier_dataframe(pareto_front, pareto_objs, algo_name, N, run_id, front_tag, pen_cap)
    csv_name = f'Frontier_{front_tag}_N{N}_Run{run_id}_{algo_name}.csv'.replace(' ', '_')
    df.to_csv(os.path.join(folder_name, csv_name), index=False)
    plot_frontier_3d(pareto_objs, f'{algo_name} | Run {run_id} | {front_tag} Frontier (3D)', os.path.join(folder_name, f'Frontier_{front_tag}_3D_N{N}_Run{run_id}_{algo_name}.png'.replace(' ', '_')))
    plot_frontier_2d(pareto_objs, f'{algo_name} | Run {run_id} | {front_tag} Frontier (Cost-Time, color=Penalty)', os.path.join(folder_name, f'Frontier_{front_tag}_2D_N{N}_Run{run_id}_{algo_name}.png'.replace(' ', '_')))
    return df

def build_run_row(N, run_id, algo_name, pareto_front, pareto_objs, rt, conv_gen, hv_ref, front_tag, pen_cap=None):
    if not pareto_objs:
        return {'N': N, 'Run': run_id, 'Algorithm': algo_name, 'Version': ALGO_VERSION, 'FrontType': front_tag, 'PenaltyCap': pen_cap, 'Pareto_Count': 0, 'HV': None, 'RunTime_s': round(rt, 6), 'Conv_Gen': conv_gen}
    mode_stats = summarize_front_modes(pareto_front, pareto_objs)
    curr_hv = hv.hypervolume(pareto_objs, hv_ref)
    return {'N': N, 'Run': run_id, 'Algorithm': algo_name, 'Version': ALGO_VERSION, 'FrontType': front_tag, 'PenaltyCap': pen_cap, 'Pareto_Count': len(pareto_objs), 'HV': round(curr_hv, 6), 'RunTime_s': round(rt, 6), 'Conv_Gen': conv_gen, 'Cost_Min': round(min((o[0] for o in pareto_objs)), 6), 'Cost_Mean': round(float(np.mean([o[0] for o in pareto_objs])), 6), 'Cost_Max': round(max((o[0] for o in pareto_objs)), 6), 'Time_Min': round(min((o[1] for o in pareto_objs)), 6), 'Time_Mean': round(float(np.mean([o[1] for o in pareto_objs])), 6), 'Time_Max': round(max((o[1] for o in pareto_objs)), 6), 'Pen_Min': round(min((o[2] for o in pareto_objs)), 6), 'Pen_Mean': round(float(np.mean([o[2] for o in pareto_objs])), 6), 'Pen_Max': round(max((o[2] for o in pareto_objs)), 6), 'TruckService_Avg': mode_stats['TruckService_Avg'], 'DirectService_Avg': mode_stats['DirectService_Avg'], 'CarService_Avg': mode_stats['CarService_Avg'], 'AirService_Avg': mode_stats['AirService_Avg'], 'ActiveTruck_Avg': mode_stats['ActiveTruck_Avg'], 'ActiveDirect_Avg': mode_stats['ActiveDirect_Avg'], 'ActiveCar_Avg': mode_stats['ActiveCar_Avg'], 'PureTruck_Ratio': mode_stats['PureTruck_Ratio'], 'ZeroPenalty_Count': mode_stats['ZeroPenalty_Count'], 'ZeroPenalty_Ratio': mode_stats['ZeroPenalty_Ratio'], 'ZeroPenaltyTime_Best': mode_stats['ZeroPenaltyTime_Best'], 'ZeroPenaltyCost_Best': mode_stats['ZeroPenaltyCost_Best']}

def build_scale_summary(run_df, front_tag):
    rows = []
    if run_df.empty:
        return pd.DataFrame(rows)
    for algo_name in sorted(run_df['Algorithm'].unique()):
        sub = run_df[run_df['Algorithm'] == algo_name]
        rows.append({'N': int(sub['N'].iloc[0]), 'Algorithm': algo_name, 'Version': ALGO_VERSION, 'FrontType': front_tag, 'Cost_Min_Best': round(sub['Cost_Min'].min(), 6), 'Cost_Min_Mean': round(sub['Cost_Min'].mean(), 6), 'Cost_Min_Std': round(sub['Cost_Min'].std(ddof=0), 6), 'Cost_Min_Worst': round(sub['Cost_Min'].max(), 6), 'Time_Min_Best': round(sub['Time_Min'].min(), 6), 'Time_Min_Mean': round(sub['Time_Min'].mean(), 6), 'Time_Min_Std': round(sub['Time_Min'].std(ddof=0), 6), 'Time_Min_Worst': round(sub['Time_Min'].max(), 6), 'Pen_Min_Best': round(sub['Pen_Min'].min(), 6), 'Pen_Min_Mean': round(sub['Pen_Min'].mean(), 6), 'Pen_Min_Std': round(sub['Pen_Min'].std(ddof=0), 6), 'Pen_Min_Worst': round(sub['Pen_Min'].max(), 6), 'HV_Best': round(sub['HV'].max(), 6), 'HV_Mean': round(sub['HV'].mean(), 6), 'HV_Std': round(sub['HV'].std(ddof=0), 6), 'RunTime_Mean_s': round(sub['RunTime_s'].mean(), 6), 'RunTime_Std_s': round(sub['RunTime_s'].std(ddof=0), 6), 'ParetoCount_Mean': round(sub['Pareto_Count'].mean(), 6), 'ParetoCount_Std': round(sub['Pareto_Count'].std(ddof=0), 6), 'TruckService_Avg': round(sub['TruckService_Avg'].mean(), 6), 'DirectService_Avg': round(sub['DirectService_Avg'].mean(), 6), 'CarService_Avg': round(sub['CarService_Avg'].mean(), 6), 'AirService_Avg': round(sub['AirService_Avg'].mean(), 6), 'ActiveTruck_Avg': round(sub['ActiveTruck_Avg'].mean(), 6), 'ActiveDirect_Avg': round(sub['ActiveDirect_Avg'].mean(), 6), 'ActiveCar_Avg': round(sub['ActiveCar_Avg'].mean(), 6), 'PureTruck_Ratio': round(sub['PureTruck_Ratio'].mean(), 6), 'ZeroPenalty_Count_Mean': round(sub['ZeroPenalty_Count'].mean(), 6), 'ZeroPenalty_Ratio_Mean': round(sub['ZeroPenalty_Ratio'].mean(), 6)})
    return pd.DataFrame(rows)


def run_traditional_nsga2(seed_value, n_customers, pop_size=100, n_gen=100):
    return run_nsga2_pure(seed_value, n_customers, init_population_random, False, n_gen=n_gen, pop_size=pop_size)


def run_kmeans_initialized_nsga2(seed_value, n_customers, pop_size=100, n_gen=100):
    return run_nsga2_pure(seed_value, n_customers, init_population_kmeans_std, True, n_gen=n_gen, pop_size=pop_size)


def run_improved_nsga2(seed_value, n_customers, pop_size=100, n_gen=100):
    return run_nsga2_pure(seed_value, n_customers, init_population_hpp, True, n_gen=n_gen, pop_size=pop_size)


def run_discrete_mopso(seed_value, n_customers, pop_size=100, n_gen=100):
    random.seed(seed_value)
    np.random.seed(seed_value)
    start_time = time.time()
    particles = init_population_random(pop_size, n_customers)
    objectives = [evaluate_individual(p) for p in particles]
    personal_best = [list(p) for p in particles]
    personal_best_objectives = list(objectives)
    front0 = fast_nondominated_sort(normalize_objs(objectives))[0]
    archive = [particles[i] for i in front0]
    archive_objectives = [objectives[i] for i in front0]
    history_cost = [min(o[0] for o in objectives)]
    convergence_gen = n_gen
    for gen in range(n_gen):
        for i, particle in enumerate(particles):
            leader = random.choice(archive) if archive else personal_best[i]
            if random.random() < 0.4:
                particle, _ = crossover(particle, personal_best[i])
            if random.random() < 0.4:
                particle, _ = crossover(particle, leader)
            if random.random() < 0.1:
                particle, = random_mode_mutation(particle)
            particles[i] = repair_individual(particle)
        objectives = [evaluate_individual(p) for p in particles]
        for i in range(pop_size):
            if dominates(objectives[i], personal_best_objectives[i]):
                personal_best[i] = list(particles[i])
                personal_best_objectives[i] = objectives[i]
        archive.extend([list(p) for p in particles])
        archive_objectives.extend(objectives)
        archive, archive_objectives = select_nsga2(archive, archive_objectives, pop_size)
        history_cost.append(min(o[0] for o in archive_objectives))
        if gen >= 40 and convergence_gen == n_gen:
            if abs(history_cost[-1] - history_cost[-40]) < 0.05:
                convergence_gen = gen
                break
    return archive, archive_objectives, time.time() - start_time, history_cost, convergence_gen


def get_algorithm_map():
    return {
        'Improved NSGA-II': run_improved_nsga2,
        'Traditional NSGA-II': run_traditional_nsga2,
        'KMeans-initialized NSGA-II': run_kmeans_initialized_nsga2,
        'Discrete MOPSO': run_discrete_mopso,
    }
