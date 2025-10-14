import os
import json
import numpy as np
import itertools

# tính ma trận thời gian di chuyển 
def compute_travel_time(cluster_heads, positions, v_AUV=10.0):
    n = len(cluster_heads)
    travel_time = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(positions[cluster_heads[i]] - positions[cluster_heads[j]])
                travel_time[i][j] = dist / v_AUV
    return travel_time

# chọn đường đi 
def path_selection(cluster_heads, travel_time, positions, v_AUV):
    n = len(cluster_heads)
    best_time = float('inf')
    best_path = None
    
    O = np.array([0, 0, 0])
    for path in itertools.permutations(range(n)):
        total_time = 0
        start_head_cluster = cluster_heads[path[0]]
        total_time += np.linalg.norm(positions[start_head_cluster] - O) / v_AUV
        total_time += sum(travel_time[path[i-1]][path[i]] for i in range(1, n))
        end_head = cluster_heads[path[-1]]
        total_time += np.linalg.norm(positions[end_head] - O) / v_AUV
        if total_time < best_time:
            best_time = total_time
            best_path = path
    return best_path, best_time

