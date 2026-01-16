import sys
import os
import json
import numpy as np

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from numba import njit

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from compute import Computing
from clustering import Clustering
from algorthms.greedy import Greedy
from algorthms.ga import ClusterTSP_GA
from algorthms.pso import Pso_routing
from algorthms.pso_adaptive_noise import Pso_adaptive_noise
from algorthms.pso_levy_flight import Pso_levy_flight

def compare_routing_boxplot():
    """
    So sánh 4 thuật toán (GA, PSO, PSOver2, PSOver3) trên 10 file dữ liệu
    Mỗi thuật toán chạy 5 lần lặp lại trên mỗi file
    Vẽ box plot: Ox = tên file, Oy = travel time, mỗi file có 4 box (4 thuật toán)
    Tạo 9 biểu đồ cho 9 thư mục (nodes_150, nodes_200, ..., nodes_550)
    """
    # ĐƯỜNG DẪN
    base_input_dir = "/kaggle/input/input-10-files-2/input_data_evenly_distributed"
    base_output_dir = "D:/Year 4/tiến hóa/project/results/routing_boxplot"
    
    os.makedirs(base_output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("SO SÁNH ĐỘ HỘI TỤ KẾT QUẢ ĐỊNH TUYẾN CỦA CÁC THUẬT TOÁN")
    print(f"{'='*80}\n")
    
    # Tham số
    INITIAL_ENERGY = 100.0
    v_f = 1.2
    v_AUV = 3.0
    R_SEN = 60
    MAX_SIZE = 25
    MIN_SIZE = 10
    
    # Tham số thuật toán
    MAX_ITER = 200  # Cho PSO
    MAX_GEN = 200   # Cho GA
    N_PARTICLES = 50
    POP_SIZE = 50
    N_RUNS = 5  # Số lần chạy lặp lại
    
    # Danh sách số nodes
    node_counts = [150, 200, 250, 300, 350, 400, 450, 500, 550]
    
    # Duyệt qua từng thư mục
    for N in node_counts:
        folder_name = f"nodes_{N}"
        input_folder = os.path.join(base_input_dir, folder_name)
        output_folder = os.path.join(base_output_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"XỬ LÝ THƯ MỤC: {folder_name}")
        print(f"{'='*80}\n")
        
        if not os.path.exists(input_folder):
            print(f"❌ Lỗi: Thư mục {input_folder} không tồn tại!")
            continue
        
        # Dictionary lưu kết quả: {file_name: {algorithm: [run1, run2, ..., run5]}}
        results_data = {}
        
        # Duyệt qua 10 file
        for file_idx in range(1, 11):
            filename = f"nodes_{N}_{file_idx}.json"
            filepath = os.path.join(input_folder, filename)
            
            if not os.path.exists(filepath):
                print(f"⚠️ Bỏ qua: {filename} không tồn tại")
                continue
            
            print(f"\n{'─'*60}")
            print(f"📂 File: {filename}")
            print(f"{'─'*60}")
            
            # Đọc dữ liệu
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"❌ Lỗi đọc file: {e}")
                continue
            
            # Parse nodes
            node_positions = {}
            initial_nodes = {}
            
            for node in data:
                nid = node['id']
                initial_nodes[nid] = {
                    'initial_energy': node.get('energy_node', INITIAL_ENERGY),
                    'residual_energy': node.get('energy_residual', INITIAL_ENERGY)
                }
                node_positions[nid] = (node['x'], node['y'], node['z'])
            
            # Phân cụm
            clustering = Clustering(space_size=400, r_sen=R_SEN, 
                                   max_cluster_size=MAX_SIZE, min_cluster_size=MIN_SIZE)
            ids = sorted(list(initial_nodes.keys()))
            coords = np.array([node_positions[nid] for nid in ids])
            clusters_data = clustering.cluster_with_constraints(coords, ids)
            
            clusters = {}
            for i, (cluster_nodes, cluster_ids) in enumerate(clusters_data):
                center = np.mean(cluster_nodes, axis=0).tolist()
                ch = clustering.choose_cluster_head(cluster_nodes, cluster_ids, initial_nodes)
                clusters[i] = {'nodes': cluster_ids, 'center': center, 'cluster_head': ch}
            
            # Tạo tọa độ routing (BS + Cluster Heads)
            sorted_keys = sorted(clusters.keys())
            centers = [(200, 200, 400)]  # Base Station
            for k in sorted_keys:
                ch = clusters[k]['cluster_head']
                centers.append(tuple(node_positions[ch]))
            center_coords = np.array(centers)
            
            print(f"  📊 Nodes: {len(initial_nodes)}, Clusters: {len(clusters)}, Routing points: {len(center_coords)}")
            
            # Khởi tạo dictionary cho file này
            results_data[filename] = {
                'GA': [],
                'PSO': [],
                'PSOver2': [],
                'PSOver3': []
            }
            
            # ============================================
            # CHẠY 5 LẦN CHO MỖI THUẬT TOÁN
            # ============================================
            
            # 1. GA
            print(f"  🔄 Chạy GA (5 lần)...", end=' ')
            for run in range(N_RUNS):
                ga_solver = ClusterTSP_GA(clusters, ga_params={
                    'pop_size': POP_SIZE,
                    'generations': MAX_GEN,
                    'v_f': v_f,
                    'v_AUV': v_AUV,
                    'verbose': False
                })
                _, _, cost_ga, _ = ga_solver.evolve()
                results_data[filename]['GA'].append(cost_ga)
            print(f"✓ (avg: {np.mean(results_data[filename]['GA']):.2f}s)")
            
            # 2. PSO
            print(f"  🔄 Chạy PSO (5 lần)...", end=' ')
            for run in range(N_RUNS):
                path_pso, cost_pso = Pso_routing.multi_pso_tsp(center_coords, v_f=v_f, v_AUV=v_AUV, n_outer=5,  # Số lần chạy outer loop
                verbose=False,
                n_particles=N_PARTICLES,  
                max_iter=MAX_ITER  
                )
                results_data[filename]['PSO'].append(cost_pso)
            print(f"✓ (avg: {np.mean(results_data[filename]['PSO']):.2f}s)")
            
            # 3. PSOver2
            print(f"  🔄 Chạy PSOver2 (5 lần)...", end=' ')
            for run in range(N_RUNS):
                path_pso2, cost_pso2 = Pso_adaptive_noise.multi_pso_tsp(
                    center_coords, 
                    v_f=v_f, 
                    v_AUV=v_AUV, 
                    n_outer=5,  # Số lần chạy outer loop
                    verbose=False,
                    n_particles=N_PARTICLES,  # Truyền vào kwargs
                    max_iter=MAX_ITER  # Truyền vào kwargs
                    )
                results_data[filename]['PSOver2'].append(cost_pso2)
            print(f"✓ (avg: {np.mean(results_data[filename]['PSOver2']):.2f}s)")
            
            # 4. PSOver3
            print(f"  🔄 Chạy PSOver3 (5 lần)...", end=' ')
            for run in range(N_RUNS):
                path_pso3, cost_pso3 = Pso_levy_flight.multi_pso_tsp(
                    center_coords, 
                    v_f=v_f, 
                    v_AUV=v_AUV, 
                    n_outer=5,  # Số lần chạy outer loop
                    verbose=False,
                    n_particles=N_PARTICLES,  # Truyền vào kwargs
                    max_iter=MAX_ITER  # Truyền vào kwargs
                    )
                results_data[filename]['PSOver3'].append(cost_pso3)
            print(f"✓ (avg: {np.mean(results_data[filename]['PSOver3']):.2f}s)")
        
        # ============================================
        # VẼ BOX PLOT
        # ============================================
        print(f"\n{'='*60}")
        print("📊 VẼ BOX PLOT")
        print(f"{'='*60}\n")
        
        fig, ax = plt.subplots(figsize=(20, 8))
        
        # Tên các file (trục X)
        file_names = [f"nodes_{N}_{i}" for i in range(1, 11)]
        x_positions = np.arange(len(file_names))
        
        # Cấu hình cho 4 thuật toán
        algorithms = ['GA', 'PSO', 'PSOver2', 'PSOver3']
        colors = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#95E1D3']
        markers = ['o', 's', '^', 'D']  # Hình dạng marker khác nhau
        linestyles = ['-', '--', '-.', ':']  # Kiểu đường thẳng khác nhau
        
        # Độ rộng offset cho mỗi thuật toán
        offset_width = 0.15
        offsets = [-1.5 * offset_width, -0.5 * offset_width, 
                   0.5 * offset_width, 1.5 * offset_width]
        
        # Vẽ cho mỗi thuật toán
        for alg_idx, (alg, color, marker, linestyle, offset) in enumerate(
            zip(algorithms, colors, markers, linestyles, offsets)
        ):
            for file_idx, filename in enumerate(file_names):
                full_filename = f"{filename}.json"
                
                if full_filename not in results_data:
                    continue
                
                values = results_data[full_filename][alg]
                
                if len(values) == 0:
                    continue
                
                x_pos = x_positions[file_idx] + offset
                
                # Vẽ đường thẳng dọc từ min đến max
                min_val = min(values)
                max_val = max(values)
                ax.plot([x_pos, x_pos], [min_val, max_val], 
                       color=color, linestyle=linestyle, linewidth=2, alpha=0.7)
                
                # Vẽ các điểm dữ liệu
                ax.scatter([x_pos] * len(values), values, 
                          color=color, marker=marker, s=80, 
                          edgecolors='black', linewidths=1, 
                          alpha=0.8, zorder=3)
                
                # Vẽ median (đường ngang)
                median_val = np.median(values)
                ax.plot([x_pos - 0.03, x_pos + 0.03], [median_val, median_val], 
                       color='black', linewidth=3, zorder=4)
        
        # Tạo legend
        legend_elements = [
            plt.Line2D([0], [0], color=color, marker=marker, linestyle=linestyle,
                      markersize=8, linewidth=2, label=alg)
            for alg, color, marker, linestyle in zip(algorithms, colors, markers, linestyles)
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=12, 
                 framealpha=0.9, edgecolor='black')
        
        # Cấu hình trục
        ax.set_xticks(x_positions)
        ax.set_xticklabels(file_names, rotation=45, ha='right', fontsize=11)
        ax.set_xlabel('Dataset Files', fontweight='bold', fontsize=14)
        ax.set_ylabel('Travel Time (s)', fontweight='bold', fontsize=14)
        ax.set_title(f'Routing Algorithm Comparison - {folder_name} (5 runs per algorithm)', 
                    fontweight='bold', fontsize=16)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        chart_file = os.path.join(output_folder, f'boxplot_{folder_name}.png')
        plt.savefig(chart_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Biểu đồ đã lưu: {chart_file}\n")
        
        # ============================================
        # LƯU KẾT QUẢ JSON
        # ============================================
        summary = {
            'folder': folder_name,
            'num_nodes': N,
            'num_files': len(results_data),
            'num_runs_per_algorithm': N_RUNS,
            'results': {}
        }
        
        for filename in file_names:
            full_filename = f"{filename}.json"
            if full_filename in results_data:
                summary['results'][filename] = {}
                for alg in algorithms:
                    values = results_data[full_filename][alg]
                    if len(values) > 0:
                        summary['results'][filename][alg] = {
                            'values': [float(v) for v in values],
                            'mean': float(np.mean(values)),
                            'median': float(np.median(values)),
                            'std': float(np.std(values)),
                            'min': float(min(values)),
                            'max': float(max(values))
                        }
        
        results_file = os.path.join(output_folder, f'results_{folder_name}.json')
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"✅ Kết quả JSON đã lưu: {results_file}\n")
        
        # ============================================
        # IN BẢNG THỐNG KÊ
        # ============================================
        print(f"{'='*90}")
        print(f"BẢNG THỐNG KÊ - {folder_name}")
        print(f"{'='*90}")
        print(f"{'File':<15} {'Algorithm':<10} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 90)
        
        for filename in file_names:
            full_filename = f"{filename}.json"
            if full_filename in results_data:
                for alg in algorithms:
                    values = results_data[full_filename][alg]
                    if len(values) > 0:
                        print(f"{filename:<15} {alg:<10} {np.mean(values):<10.2f} "
                              f"{np.median(values):<10.2f} {np.std(values):<10.2f} "
                              f"{min(values):<10.2f} {max(values):<10.2f}")
                print("-" * 90)
        
        print(f"\n✅ Hoàn thành thư mục {folder_name}!\n")
    
    print(f"\n{'='*80}")
    print(f"✅ HOÀN THÀNH TẤT CẢ! Kết quả tại: {base_output_dir}")
    print(f"{'='*80}\n")
print("✓ Box plot comparison code loaded")

compare_routing_boxplot()