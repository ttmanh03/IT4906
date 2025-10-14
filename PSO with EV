import numpy as np
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

# TÍNH Vs
def compute_vs(p1, p2, v_f, v_AUV):
    """Tính vận tốc tổng hợp v_s giữa 2 vị trí p1, p2"""
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    Lx, Ly, Lz = x2 - x1, y2 - y1, z2 - z1
    L_mag = math.sqrt(Lx**2 + Ly**2 + Lz**2)
    if L_mag == 0:
        return v_AUV

    cos_beta = Lz / L_mag
    cos_beta = np.clip(cos_beta, -1, 1)
    beta = math.acos(cos_beta)

    if abs(cos_beta) < 1e-6:
        cos_beta = 1e-6

    inner = (v_f * cos_beta) / v_AUV
    inner = np.clip(inner, -1, 1)
    angle = beta + math.acos(inner)
    v_s = abs(math.cos(angle) * v_AUV / cos_beta)

    return v_s


# TÍNH THỜI GIAN DI CHUYỂN
def travel_time(path, coords, v_f, v_AUV):
    total_time = 0.0
    for i in range(len(path) - 1):
        p1 = coords[path[i]]
        p2 = coords[path[i + 1]]
        d = np.linalg.norm(p2 - p1)
        v_s = compute_vs(p1, p2, v_f, v_AUV)
        total_time += d / v_s
    p1 = coords[path[-1]]
    p2 = coords[path[0]]
    d = np.linalg.norm(p2 - p1)
    v_s = compute_vs(p1, p2, v_f, v_AUV)
    total_time += d / v_s
    return total_time


# VẬN TỐC SWARM
def get_swap_sequence(A, B):
    seq = []
    temp = A.copy()
    for i in range(1, len(A)):
        if temp[i] != B[i]:
            j = temp.index(B[i])
            seq.append((i, j))
            temp[i], temp[j] = temp[j], temp[i]
    return seq


def apply_velocity(position, velocity):
    new_pos = position.copy()
    for (i, j) in velocity:
        if i == 0 or j == 0:
            continue
        new_pos[i], new_pos[j] = new_pos[j], new_pos[i]
    return new_pos


# PSO CHO TSP
def pso_tsp_3d_time(coords, v_f=1.0, v_AUV=3.0, n_particles=40, max_iter=200, w=0.7, c1=0.5, c2=0.5, init_gbest=None):
    n_cities = len(coords)
    cities = list(range(1, n_cities))
    swarm = [[0] + random.sample(cities, len(cities)) for _ in range(n_particles)]
    velocities = [[] for _ in range(n_particles)]

    costs = [travel_time(p, coords, v_f, v_AUV) for p in swarm]
    pbest = list(swarm)
    pbest_cost = list(costs)

    if init_gbest is not None:
        gbest = init_gbest.copy()
        gbest_cost = travel_time(gbest, coords, v_f, v_AUV)
    else:
        gbest = pbest[np.argmin(pbest_cost)]
        gbest_cost = min(pbest_cost)

    for t in range(max_iter):
        for i in range(n_particles):
            xi = swarm[i]
            vi = velocities[i]

            v_new = []
            n_keep = int(w * len(vi))
            v_new.extend(vi[:n_keep])

            if random.random() < c1:
                seq_pb = get_swap_sequence(xi, pbest[i])
                if seq_pb:
                    v_new.extend(random.sample(seq_pb, k=min(len(seq_pb), 2)))

            if random.random() < c2:
                seq_gb = get_swap_sequence(xi, gbest)
                if seq_gb:
                    v_new.extend(random.sample(seq_gb, k=min(len(seq_gb), 2)))

            velocities[i] = v_new
            new_x = apply_velocity(xi, v_new)
            swarm[i] = new_x
            new_cost = travel_time(new_x, coords, v_f, v_AUV)

            if new_cost < pbest_cost[i]:
                pbest[i] = new_x
                pbest_cost[i] = new_cost
                if new_cost < gbest_cost:
                    gbest = new_x
                    gbest_cost = new_cost

        w = 0.7 - 0.5 * (t / max_iter)
        if t % 20 == 0:
            print(f"  Inner iter {t}: Best time = {gbest_cost:.4f}")

    return gbest, gbest_cost


# PSO ĐA VÒNG
def multi_pso_tsp(coords, v_f=1.2, v_AUV=3.0, n_outer=20, **kwargs):
    prev_gbest = None
    prev_cost = None

    for outer in range(n_outer):
        print(f"\n=== Outer loop {outer + 1}/{n_outer} ===")
        gbest, cost = pso_tsp_3d_time(coords, v_f=v_f, v_AUV=v_AUV, init_gbest=prev_gbest, **kwargs)
        print(f"  → Best path time = {cost:.4f}")

        # Dừng nếu không thay đổi
        if prev_cost is not None and abs(cost - prev_cost) < 1e-6:
            print("  → Converged! Stop early.")
            break

        prev_gbest = gbest
        prev_cost = cost

    return prev_gbest, prev_cost


# TÍNH NĂNG LƯỢNG
def compute_energy(best_time, G=100, L=1024, n=4,
                   P_t=1.6e-3, P_r=0.8e-3, P_idle=0.1e-3,
                   DR=4000, DR_i=1e6):
    """Tính năng lượng tiêu thụ cho Member Node và Target Node"""
    
    # --- Member Node ---
    E_tx_MN = G * P_t * L / DR
    E_idle_MN = (best_time - G * L / DR) * P_idle
    E_total_MN = E_tx_MN + E_idle_MN

    # --- Target Node ---
    E_rx_TN = G * P_r * L * n / DR
    E_tx_TN = G * P_t * L * n / DR_i
    E_idle_TN = (best_time - (G * L * n / DR) - (G * L * n / DR_i)) * P_idle
    E_total_TN = E_rx_TN + E_tx_TN + E_idle_TN

    return {
        "Member": {"E_tx": E_tx_MN, "E_idle": E_idle_MN, "E_total": E_total_MN},
        "Target": {"E_rx": E_rx_TN, "E_tx": E_tx_TN, "E_idle": E_idle_TN, "E_total": E_total_TN}
    }


# VẼ ĐƯỜNG ĐI 3D
def plot_path_3d(coords, path, title="Best path"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    path_coords = coords[path + [path[0]]]
    ax.plot(path_coords[:, 0], path_coords[:, 1], path_coords[:, 2], "-o")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c="red", s=50)
    for i, (x, y, z) in enumerate(coords):
        ax.text(x, y, z, str(i))
    ax.set_title(title)
    plt.show()


# TEST
if __name__ == "__main__":
    with open("/kaggle/input/inputdata-test/nodes_100.json", "r") as f:
        nodes_data = json.load(f)
    coords = np.array([cluster["center"] for cluster in nodes_data.values()])

    print(f"Đã đọc {len(coords)} nodes từ file JSON.")
    best_path, best_time = multi_pso_tsp(coords, v_f=1, v_AUV=2.0, n_particles=30, max_iter=200)

    print("\nFinal best path:", best_path)
    print(f"Final best total travel time: {best_time:.4f}s")

    # --- TÍNH NĂNG LƯỢNG ---
    energy = compute_energy(best_time)
    print("\n=== Năng lượng tiêu thụ ===")
    for k, v in energy.items():
        print(f"{k}:")
        for name, val in v.items():
            print(f"   {name} = {val:.6f} J")

    plot_path_3d(coords, best_path, title=f"PSO Multi-Loop - Min Time ({best_time:.2f}s)")
