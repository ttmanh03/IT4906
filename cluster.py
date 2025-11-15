import numpy as np
import json
import os
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

class Clustering:
    def __init__(self, space_size=400, r_sen=50, max_cluster_size=20, min_cluster_size=5):
        self.space_size = space_size
        self.r_sen = r_sen
        self.max_cluster_size = max_cluster_size
        self.min_cluster_size = min_cluster_size

    def estimate_optimal_k(self, nodes, base_station=(200,200,400)):
        """
        Ước tính số cụm tối ưu dựa trên công thức WSN
        K = sqrt(N*L / (pi*d_tobs))
        """
        N = len(nodes)
        base_pos = np.array(base_station)

        # Khoang cach trung binh toi base station
        distances = np.linalg.norm(nodes - base_pos, axis=1)
        d_tobs = np.mean(distances)

        space_size = self.space_size

        k_optimal = np.sqrt(N * space_size / (np.pi * d_tobs))
        k_optimal = max(2, int(np.round(k_optimal)))

        # Điều chỉnh dựa trên max_cluster_size
        k_min = int(np.ceil(N / self.max_cluster_size))
        k_optimal = max(k_optimal, k_min)
        
        return k_optimal
    
    def check_cluster_validity(self, cluster_nodes):
        """
        Kiem tra tinh hop le cua cum
        """
        size = len(cluster_nodes)

        # Kiểm tra kích thước
        if size < self.min_cluster_size or size > self.max_cluster_size:
            return False, 0, size
        
        # Kiểm tra khoảng cách
        if size > 1:
            distances = pdist(cluster_nodes)
            max_dist = np.max(distances)
            
            if max_dist > self.r_sen:
                return False, max_dist, size
            
            return True, max_dist, size
        
        return True, 0, size
    
    def split_invalid_cluster(self, cluster_nodes, cluster_ids):
        """
        Chia nhỏ cụm không hợp lệ thành các cụm con
        """
        # Sử dụng K-Means để chia 2
        kmeans = KMeans(n_clusters=2, n_init=20, random_state=42)
        labels = kmeans.fit_predict(cluster_nodes)
        
        sub_clusters = []
        for i in range(2):
            sub_nodes = cluster_nodes[labels == i]
            sub_ids = [cluster_ids[j] for j in range(len(cluster_ids)) if labels[j] == i]
            
            if len(sub_nodes) > 0:
                sub_clusters.append((sub_nodes, sub_ids))
        
        return sub_clusters
    
    def merge_small_clusters(self, clusters_data):
        """
        Gộp các cụm nhỏ với cụm láng giềng gần nhất
        """
        if len(clusters_data) <= 1:
            return clusters_data
        
        merged = []
        to_merge = []
        
        # Tìm các cụm nhỏ
        for nodes, ids in clusters_data:
            if len(nodes) < self.min_cluster_size:
                to_merge.append((nodes, ids))
            else:
                merged.append((nodes, ids))
        
        # Gộp từng cụm nhỏ vào cụm gần nhất
        for small_nodes, small_ids in to_merge:
            if len(merged) == 0:
                merged.append((small_nodes, small_ids))
                continue
            
            # Tìm cụm gần nhất
            small_center = np.mean(small_nodes, axis=0)
            min_dist = float('inf')
            best_idx = 0
            
            for i, (nodes, ids) in enumerate(merged):
                center = np.mean(nodes, axis=0)
                dist = np.linalg.norm(small_center - center)
                
                # Kiểm tra xem gộp có vượt quá max_size không
                if dist < min_dist and len(nodes) + len(small_nodes) <= self.max_cluster_size:
                    min_dist = dist
                    best_idx = i
            
            # Gộp
            merged[best_idx] = (
                np.vstack([merged[best_idx][0], small_nodes]),
                merged[best_idx][1] + small_ids
            )
        
        return merged
    
    def cluster_with_constraints(self, nodes, node_ids, k=None, max_iterations=10):
        """
        Phân cụm với ràng buộc - Thuật toán chính
        
        Args:
            nodes: Tọa độ 3D của nodes
            node_ids: ID của nodes
            k: Số cụm (nếu None sẽ tự động ước tính)
            max_iterations: Số lần lặp tối đa để điều chỉnh
            
        Returns:
            List of (cluster_nodes, cluster_ids)
        """
        if k is None:
            k = self.estimate_optimal_k(nodes)
        
        print(f"Bắt đầu phân cụm với k={k}")
        
        # Bước 1: K-Means ban đầu
        kmeans = KMeans(n_clusters=k, n_init=30, random_state=42)
        labels = kmeans.fit_predict(nodes)
        
        # Bước 2: Tạo các cụm và kiểm tra
        iteration = 0
        while iteration < max_iterations:
            print(f"  Vòng lặp {iteration + 1}/{max_iterations}")
            
            valid_clusters = []
            invalid_clusters = []
            
            # Phân loại cụm hợp lệ và không hợp lệ
            for i in range(k):
                cluster_nodes = nodes[labels == i]
                cluster_ids = [node_ids[j] for j in range(len(node_ids)) if labels[j] == i]
                
                if len(cluster_nodes) == 0:
                    continue
                
                is_valid, max_dist, size = self.check_cluster_validity(cluster_nodes)
                
                if is_valid:
                    valid_clusters.append((cluster_nodes, cluster_ids))
                    print(f"    Cụm {i}: ✓ hợp lệ (size={size}, max_dist={max_dist:.1f}m)")
                else:
                    invalid_clusters.append((cluster_nodes, cluster_ids))
                    print(f"    Cụm {i}: ✗ không hợp lệ (size={size}, max_dist={max_dist:.1f}m)")
            
            # Nếu tất cả hợp lệ, kết thúc
            if len(invalid_clusters) == 0:
                print(f"  → Tất cả cụm hợp lệ!")
                break
            
            # Bước 3: Xử lý các cụm không hợp lệ
            for cluster_nodes, cluster_ids in invalid_clusters:
                size = len(cluster_nodes)
                
                if size > self.max_cluster_size:
                    # Cụm quá lớn → Chia nhỏ
                    print(f"    → Chia cụm (size={size})")
                    sub_clusters = self.split_invalid_cluster(cluster_nodes, cluster_ids)
                    valid_clusters.extend(sub_clusters)
                else:
                    # Cụm có khoảng cách quá lớn → Chia nhỏ
                    print(f"    → Chia cụm (khoảng cách lớn)")
                    sub_clusters = self.split_invalid_cluster(cluster_nodes, cluster_ids)
                    valid_clusters.extend(sub_clusters)
            
            # Cập nhật labels và k cho vòng lặp tiếp theo
            k = len(valid_clusters)
            
            # Tạo lại labels từ valid_clusters
            labels = np.zeros(len(nodes), dtype=int)
            for cluster_idx, (_, cluster_ids) in enumerate(valid_clusters):
                for node_id in cluster_ids:
                    node_idx = node_ids.index(node_id)
                    labels[node_idx] = cluster_idx
            
            iteration += 1
        
        # Bước 4: Gộp các cụm quá nhỏ
        valid_clusters = self.merge_small_clusters(valid_clusters)
        
        print(f"Hoàn thành: {len(valid_clusters)} cụm")
        return valid_clusters
    
    def choose_cluster_head(self, cluster_nodes, cluster_ids, node_data=None):
        """
        Chọn cluster head
        - Ưu tiên: Node có năng lượng cao nhất
        - Dự phòng: Node gần tâm cụm nhất
        """
        if node_data:
            # Chọn theo năng lượng
            max_energy = -1
            ch_id = cluster_ids[0]
            
            for nid in cluster_ids:
                if nid in node_data and 'residual_energy' in node_data[nid]:
                    energy = node_data[nid]['residual_energy']
                    if energy > max_energy:
                        max_energy = energy
                        ch_id = nid
            
            return ch_id
        else:
            # Chọn theo khoảng cách đến tâm
            center = np.mean(cluster_nodes, axis=0)
            distances = np.linalg.norm(cluster_nodes - center, axis=1)
            min_idx = np.argmin(distances)
            return cluster_ids[min_idx]
    
    def calculate_metrics(self, clusters_data):
        """
        Tính các metric đánh giá chất lượng phân cụm
        """
        metrics = {
            'num_clusters': len(clusters_data),
            'avg_cluster_size': 0,
            'min_cluster_size': float('inf'),
            'max_cluster_size': 0,
            'avg_intra_distance': 0,
            'max_intra_distance': 0,
            'balance_score': 0  # Độ cân bằng kích thước cụm (0-1, càng cao càng tốt)
        }
        
        sizes = []
        intra_dists = []
        
        for nodes, ids in clusters_data:
            size = len(nodes)
            sizes.append(size)
            
            metrics['min_cluster_size'] = min(metrics['min_cluster_size'], size)
            metrics['max_cluster_size'] = max(metrics['max_cluster_size'], size)
            
            if size > 1:
                distances = pdist(nodes)
                intra_dists.append(np.mean(distances))
                metrics['max_intra_distance'] = max(metrics['max_intra_distance'], np.max(distances))
        
        metrics['avg_cluster_size'] = np.mean(sizes)
        metrics['avg_intra_distance'] = np.mean(intra_dists) if intra_dists else 0
        
        # Tính balance score (dựa trên coefficient of variation)
        cv = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0
        metrics['balance_score'] = 1 / (1 + cv)  # 1 = hoàn toàn cân bằng
        
        return metrics

def process_data(input_file, output_folder, draw_folder, 
                    r_sen=50, max_size=20, min_size=5):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    nodes = np.array([[d['x'], d['y'], d['z']] for d in data])
    node_ids = [d['id'] for d in data]
    
    # Tạo node_data với thông tin năng lượng
    node_data = {}
    for d in data:
        if 'residual_energy' in d:
            node_data[d['id']] = {
                'residual_energy': d['residual_energy'],
                'initial_energy': d.get('initial_energy', 100.0)
            }
    
    # Khởi tạo clustering
    clustering = Clustering(
        r_sen=r_sen,
        max_cluster_size=max_size,
        min_cluster_size=min_size
    )
    
    # Phân cụm
    print(f"\n{'='*60}")
    print(f"Xử lý: {os.path.basename(input_file)}")
    print(f"Số nodes: {len(nodes)}")
    print(f"Tham số: r_sen={r_sen}m, max_size={max_size}, min_size={min_size}")
    print(f"{'='*60}")
    
    clusters_data = clustering.cluster_with_constraints(nodes, node_ids)
    
    # Tính metrics
    metrics = clustering.calculate_metrics(clusters_data)
    
    print(f"\n{'='*60}")
    print("KẾT QUẢ:")
    print(f"  Số cụm: {metrics['num_clusters']}")
    print(f"  Kích thước trung bình: {metrics['avg_cluster_size']:.1f}")
    print(f"  Kích thước: [{metrics['min_cluster_size']} - {metrics['max_cluster_size']}]")
    print(f"  Khoảng cách trung bình trong cụm: {metrics['avg_intra_distance']:.1f}m")
    print(f"  Khoảng cách max trong cụm: {metrics['max_intra_distance']:.1f}m")
    print(f"  Độ cân bằng: {metrics['balance_score']:.2%}")
    print(f"{'='*60}\n")
    
    # Tạo output
    output_data = {}
    for i, (cluster_nodes, cluster_ids) in enumerate(clusters_data):
        ch = clustering.choose_cluster_head(cluster_nodes, cluster_ids, node_data)
        center = np.mean(cluster_nodes, axis=0)
        
        output_data[i] = {
            'nodes': cluster_ids,
            'center': [float(x) for x in np.round(center, 2)],
            'cluster_head': int(ch),
            'size': len(cluster_ids)
        }
    
    # Lưu output
    output_file = os.path.join(output_folder, os.path.basename(input_file))
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"✓ Đã lưu: {output_file}")
    
    # Vẽ biểu đồ
    draw_file = os.path.join(draw_folder, 
                            os.path.basename(input_file).replace('.json', '.png'))
    visualize_clusters(nodes, clusters_data, output_data, draw_file)
    print(f"✓ Đã vẽ: {draw_file}")
    
    return output_data, metrics

def visualize_clusters(nodes, clusters_data, output_data, save_path):
    """
    Vẽ biểu đồ 3D các cụm
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    num_clusters = len(clusters_data)
    cmap = plt.colormaps.get_cmap('tab20' if num_clusters > 10 else 'tab10')
    
    for i, (cluster_nodes, cluster_ids) in enumerate(clusters_data):
        color = cmap(i % 20)
        
        # Vẽ nodes
        ax.scatter(cluster_nodes[:, 0], cluster_nodes[:, 1], cluster_nodes[:, 2],
                  label=f'Cụm {i} ({len(cluster_ids)})',
                  color=color, alpha=0.6, s=50)
        
        # Vẽ cluster head
        ch_id = output_data[i]['cluster_head']
        ch_idx = cluster_ids.index(ch_id)
        ch_pos = cluster_nodes[ch_idx]
        ax.scatter(ch_pos[0], ch_pos[1], ch_pos[2],
                  color=color, marker='*', s=400, 
                  edgecolor='black', linewidth=1.5, zorder=100)
        
        # Vẽ tâm cụm
        center = output_data[i]['center']
        ax.scatter(center[0], center[1], center[2],
                  color=color, marker='x', s=100, linewidth=2, zorder=90)
    
    # Vẽ base station
    ax.scatter(0, 0, 0, color='red', marker='^', s=500,
              label='Base Station', edgecolor='black', linewidth=2, zorder=110)
    
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)
    ax.set_title(f'WSN Clustering - {len(nodes)} nodes, {num_clusters} cụm',
                fontsize=13, fontweight='bold')
    
    if num_clusters <= 15:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # Tạo thư mục output
    input_folder = "IT4906/input_data"
    output_folder = "output_data_optimized"
    draw_folder = "draw_output_optimized"
    
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(draw_folder, exist_ok=True)
    
    # Tham số
    R_SEN = 60 # Bán kính truyền tải (m)
    MAX_SIZE = 20  # Kích thước cụm tối đa
    MIN_SIZE = 5  # Kích thước cụm tối thiểu
    
    # Xử lý từng file
    all_metrics = {}
    
    for filename in sorted(os.listdir(input_folder)):
        if filename.startswith("nodes_") and filename.endswith(".json"):
            input_path = os.path.join(input_folder, filename)
            
            try:
                output_data, metrics = process_data(
                    input_path, output_folder, draw_folder,
                    r_sen=R_SEN, max_size=MAX_SIZE, min_size=MIN_SIZE
                )
                all_metrics[filename] = metrics
            except Exception as e:
                print(f"✗ Lỗi xử lý {filename}: {e}")
                continue
    
    # Tổng kết
    print("\n" + "="*60)
    print("TỔNG KẾT TOÀN BỘ")
    print("="*60)
    for filename, metrics in all_metrics.items():
        print(f"\n{filename}:")
        print(f"  Cụm: {metrics['num_clusters']}, "
              f"Size: {metrics['avg_cluster_size']:.1f} ± "
              f"{metrics['max_cluster_size']-metrics['min_cluster_size']}, "
              f"Balance: {metrics['balance_score']:.2%}")
