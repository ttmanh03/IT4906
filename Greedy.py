import os
import json
import numpy as np
import itertools
import math


input_folder = "output_data_kmeans"
output_folder = "output_select_path_greedy"
os.makedirs(output_folder, exist_ok=True)


# TÍNH Vs - Vận tốc tổng hợp (copy từ GA with EV)
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


# Tính ma trận thời gian di chuyển (cải tiến với compute_vs)
def compute_travel_time(cluster_heads, positions, v_f=1.2, v_AUV=3.0):
    n = len(cluster_heads)
    travel_time = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                p1 = positions[cluster_heads[i]]
                p2 = positions[cluster_heads[j]]
                
                # Tính khoảng cách
                dist = np.linalg.norm(np.array(p2) - np.array(p1))
                
                # Tính vận tốc tổng hợp với dòng chảy
                v_s = compute_vs(
                    tuple(p1) if isinstance(p1, (list, np.ndarray)) else p1,
                    tuple(p2) if isinstance(p2, (list, np.ndarray)) else p2,
                    v_f, 
                    v_AUV
                )
                
                # Tính thời gian
                if v_s <= 1e-9:
                    travel_time[i][j] = float('inf')
                else:
                    travel_time[i][j] = dist / v_s
    return travel_time


# Chọn đường đi (brute force - thử tất cả hoán vị)
def path_selection(cluster_heads, travel_time):
    n = len(cluster_heads)
    best_time = float('inf')
    best_path = None
    
    for path in itertools.permutations(range(n)):
        total_time = sum(travel_time[path[i-1]][path[i]] for i in range(1, n))
        if total_time < best_time:
            best_time = total_time
            best_path = path
    
    return best_path, best_time


# Xử lý từng file JSON
for filename in os.listdir(input_folder):
    if not filename.endswith(".json"):
        continue

    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, f"result_{filename}")

    print(f"\nĐang xử lý {filename}")

    with open(input_path, "r", encoding="utf-8") as f:
        nodes_data = json.load(f)

    # Trích danh sách cluster head và tọa độ tâm cụm
    cluster_heads = []
    positions = {}
    for cid, info in nodes_data.items():
        ch = info["cluster_head"]
        cluster_heads.append(ch)
        # Đảm bảo center là tuple
        center = info["center"]
        if isinstance(center, list):
            positions[ch] = tuple(center)
        else:
            positions[ch] = center

    # Tính ma trận thời gian di chuyển (với compute_vs)
    # Sử dụng v_f=1.2 và v_AUV=3.0 giống như GA with EV
    travel_time = compute_travel_time(cluster_heads, positions, v_f=1.2, v_AUV=3.0)

    # Tìm đường đi tối ưu
    best_path, best_time = path_selection(cluster_heads, travel_time)

    # Xuất kết quả
    result = {
        "input_file": filename,
        "cluster_heads": cluster_heads,
        "best_path": [cluster_heads[i] for i in best_path],
        "total_time": round(float(best_time), 4),
        "parameters": {
            "v_f": 1.2,
            "v_AUV": 3.0
        }
    }

    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(result, f_out, ensure_ascii=False, indent=4)

    print(f"Kết quả của {output_path}:")
    print(f"Tổng thời gian di chuyển nhỏ nhất: {best_time:.2f} s")

print("\nDone")