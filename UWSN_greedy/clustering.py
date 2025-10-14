import numpy as np
from sklearn.cluster import KMeans
import json

def cluster_split(nodes, node_ids, r_sen = 100, R = 20, max_depth=10, depth=0):
    """
    Hàm phân cụm lặp theo Algorithm 1
    nodes: tọa độ 3D của các node
    node_ids: list id tương ứng của các node
    r_sen: bán kính truyền tải tối đa của node, giả sử là 100m
    R: số lượng node tối đa trong 1 cụm, cho là 20
    max_depth: độ sâu đệ quy tối đa
    """
    center = np.mean(nodes, axis=0) # tâm cụm
    dists = np.linalg.norm(nodes - center, axis=1) # khoảng cách từ tâm đến các node
    if (len(nodes) <= R and np.all(dists <= r_sen)) or depth >= max_depth:
        return [{
            "node_ids": node_ids,
            "nodes": nodes,
            "center": center
        }]

    # chia cụm bằng KMeans với k = 2
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
    labels = kmeans.fit_predict(nodes)

    clusters = []
    for i in range(2):
        sub_nodes = nodes[labels == i]
        sub_ids = [node_ids[j] for j in range(len(node_ids)) if labels[j] == i]
        clusters += cluster_split(sub_nodes, sub_ids, r_sen, R, max_depth, depth + 1)

    return clusters


def choose_cluster_head(cluster, energies, alpha=0.5):
    # Chọn cluster head là node gần tâm cụm hơn và có năng lượng dư cao hơn
    nodes = cluster["nodes"]
    center = cluster["center"]
    node_ids = cluster["node_ids"]

    dists = np.linalg.norm(nodes - center, axis=1) #khoảng cách từng node đến tâm cụm
    energy_values = np.array([energies[idx] for idx in node_ids]) #năng lượng còn lại của mỗi node
    # Chuẩn hóa
    norm_d = dists / np.max(dists) if np.max(dists) != 0 else dists
    norm_e = energy_values / np.max(energy_values) if np.max(energy_values) != 0 else energy_values
    # Hàm mục tiêu
    f_score = alpha * (1 - norm_e) + (1 - alpha) * norm_d
    ch_index = np.argmin(f_score)
    cluster_head = node_ids[ch_index] #chọn node thỏa mãn làm tâm cụm
    return cluster_head
 
