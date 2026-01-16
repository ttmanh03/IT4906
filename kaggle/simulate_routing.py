import sys
import os
import json
import numpy as np

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from numba import njit
import itertools

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from compute import Computing
from clustering import Clustering
from algorthms.greedy import Greedy
from algorthms.ga import ClusterTSP_GA
from algorthms.pso import Pso_routing
from algorthms.pso_adaptive_noise import Pso_adaptive_noise
from algorthms.pso_levy_flight import Pso_levy_flight


def main():
    """
    Mô phỏng với 3 thuật toán: Greedy, GA, PSO
    Tạo folder tổng hợp và folder chi tiết cho từng bộ dữ liệu
    """
    # ĐIỀU CHỈNH ĐƯỜNG DẪN
    input_folder = "/kaggle/input/input652/input_data"
    output_main_folder = "/kaggle/working/results_history"
    
    os.makedirs(output_main_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        print(f"❌ Lỗi: Thư mục {input_folder} không tồn tại!")
        return

    files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    if len(files) == 0:
        print(f"❌ Không tìm thấy file dữ liệu nào trong {input_folder}")
        return

    # Tham số
    INITIAL_ENERGY = 100.0
    v_f = 1.2
    v_AUV = 3.0
    R_SEN = 60
    MAX_SIZE = 25
    MIN_SIZE = 10
    
    # Lưu kết quả cho 3 thuật toán (dùng cho biểu đồ tổng hợp)
    algorithms = [ 'PSO', 'Greedy', 'GA']
    global_results = {alg: {} for alg in algorithms}
    
    clustering = Clustering(space_size=400, r_sen=R_SEN, max_cluster_size=MAX_SIZE, min_cluster_size=MIN_SIZE)

    # Chạy từng file dữ liệu
    for filename in sorted(files):
        input_path = os.path.join(input_folder, filename)
        
        # Tạo folder cho file này
        base_name = filename.replace('.json', '')
        file_output_folder = os.path.join(output_main_folder, f"folder_{base_name}")
        os.makedirs(file_output_folder, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"=== Đang xử lý file: {filename} ===")
        print(f"{'='*80}")
        
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Lỗi đọc file {filename}: {e}")
            continue

        # Parse dữ liệu node
        node_positions = {}
        initial_nodes = {}
        
        if isinstance(data, list):
            for node in data:
                nid = node['id']
                initial_nodes[nid] = {
                    'initial_energy': node.get('initial_energy', INITIAL_ENERGY),
                    'residual_energy': node.get('residual_energy', INITIAL_ENERGY)
                }
                node_positions[nid] = (node['x'], node['y'], node['z'])
        else:
            print(f"❌ Cấu trúc file {filename} không được hỗ trợ")
            continue

        total_nodes = len(initial_nodes)
        print(f"Tổng số node: {total_nodes}")

        # Phân cụm ban đầu (dùng chung cho cả 3 thuật toán)
        ids = sorted(list(initial_nodes.keys()))
        coords = np.array([node_positions[nid] for nid in ids])
        clusters_data = clustering.cluster_with_constraints(coords, ids)
        
        initial_clusters = {}
        for i, (cluster_nodes, cluster_ids) in enumerate(clusters_data):
            center = np.mean(cluster_nodes, axis=0).tolist()
            ch = clustering.choose_cluster_head(cluster_nodes, cluster_ids, initial_nodes)
            initial_clusters[i] = {'nodes': cluster_ids, 'center': center, 'cluster_head': ch}

        # Tạo tọa độ cho routing
        sorted_keys = sorted(initial_clusters.keys())
        centers = [(200, 200, 400)]  # BS
        for k in sorted_keys:
            ch = initial_clusters[k]['cluster_head']
            centers.append(tuple(node_positions[ch]))
        center_coords = np.array(centers)

        # Lưu kết quả từng thuật toán cho file này
        file_results = {}

        # ======== CHẠY 3 THUẬT TOÁN ========
        for algorithm in algorithms:
            print(f"\n{'='*60}")
            print(f"🔥 Chạy thuật toán: {algorithm}")
            print(f"{'='*60}")
            
            # Deep copy dữ liệu cho mỗi thuật toán
            all_nodes = {k: v.copy() for k, v in initial_nodes.items()}
            clusters = {k: v.copy() for k, v in initial_clusters.items()}
            
            # Tính đường đi ban đầu
            if algorithm == 'Greedy':
                current_path, current_time = Greedy.greedy_tsp(center_coords, v_f, v_AUV)
            elif algorithm == 'GA':
                ga_solver = ClusterTSP_GA(clusters, ga_params={
                    'pop_size': 50,
                    'generations': 150,
                    'v_f': v_f,
                    'v_AUV': v_AUV,
                    'verbose': False
                })
                current_path, _, current_time = ga_solver.evolve()
            else:  # PSO
                current_path, current_time = Pso_routing.multi_pso_tsp(
                    center_coords, v_f, v_AUV,
                    n_outer=3, n_particles=50, max_iter=50, verbose=False
                )
            
            print(f"   Đường đi ban đầu: {current_time:.4f}s")
            
            # Mô phỏng từng chu kỳ
            cycle = 0
            alive_log = []
            total_energy_consumed = 0
            
            while True:
                cycle += 1
                alive_log.append(len(all_nodes))
                alive_ratio = len(all_nodes) / total_nodes
                
                if alive_ratio < 0.1:
                    print(f"🛑 Dừng ở cycle {cycle}: {alive_ratio*100:.2f}% node còn sống")
                    break
                
                if cycle % 50 == 0:
                    print(f"   Cycle {cycle}: {len(all_nodes)}/{total_nodes} nodes alive")
                
                # Cập nhật năng lượng
                energy_before = sum(all_nodes[n]['residual_energy'] for n in all_nodes)
                Computing.update_energy(all_nodes, clusters, current_time)
                energy_after = sum(all_nodes[n]['residual_energy'] for n in all_nodes)
                total_energy_consumed += (energy_before - energy_after)
                
                # Chọn lại cluster head
                clusters = Clustering.reselect_cluster_heads(clusters, all_nodes)
                
                # Tính lại đường đi với CH mới
                sorted_keys = sorted(clusters.keys())
                centers = [(200, 200, 400)]
                for k in sorted_keys:
                    ch = clusters[k]['cluster_head']
                    centers.append(tuple(node_positions[ch]))
                center_coords = np.array(centers)
                
                if algorithm == 'Greedy':
                    current_path, current_time = Greedy.greedy_tsp(center_coords, v_f, v_AUV)
                elif algorithm == 'GA':
                    ga_solver = ClusterTSP_GA(clusters, ga_params={
                        'pop_size': 50, 'generations': 150,
                        'v_f': v_f, 'v_AUV': v_AUV, 'verbose': False
                    })
                    current_path, _, current_time = ga_solver.evolve()
                else:  # PSO
                    current_path, current_time = Pso_routing.multi_pso_tsp(
                        center_coords, v_f, v_AUV,
                        n_outer=3, n_particles=30, max_iter=80, verbose=False
                    )
                
                # Kiểm tra node chết
                clusters, dead_nodes = Clustering.remove_dead_nodes(all_nodes, clusters)
                
                if dead_nodes:
                    if len(all_nodes) > 0:
                        clusters = Clustering.recluster(all_nodes, node_positions, clustering, R_SEN, MAX_SIZE, MIN_SIZE)
                        if len(clusters) == 0:
                            break
                        
                        sorted_keys = sorted(clusters.keys())
                        centers = [(200, 200, 400)]
                        for k in sorted_keys:
                            ch = clusters[k]['cluster_head']
                            centers.append(tuple(node_positions[ch]))
                        center_coords = np.array(centers)
                        
                        if algorithm == 'Greedy':
                            current_path, current_time = Greedy.greedy_tsp(center_coords, v_f, v_AUV)
                        elif algorithm == 'GA':
                            ga_solver = ClusterTSP_GA(clusters, ga_params={
                                'pop_size': 50, 'generations': 150,
                                'v_f': v_f, 'v_AUV': v_AUV, 'verbose': False
                            })
                            current_path, _, current_time = ga_solver.evolve()
                        else:  # PSO
                            current_path, current_time = Pso_routing.multi_pso_tsp(
                                center_coords, v_f, v_AUV,
                                n_outer=3, n_particles=30, max_iter=80, verbose=False
                            )
                    else:
                        break
            
            # Lưu kết quả thuật toán này
            final_alive = len(all_nodes)
            final_alive_ratio = final_alive / total_nodes
            
            result_data = {
                'filename': filename,
                'algorithm': algorithm,
                'initial_nodes': total_nodes,
                'cycles_completed': cycle - 1,
                'final_alive_nodes': final_alive,
                'final_alive_ratio': round(final_alive_ratio, 4),
                'total_energy_consumed': round(total_energy_consumed, 4),
                'alive_log': alive_log
            }
            
            file_results[algorithm] = result_data
            
            # Lưu file JSON riêng cho thuật toán này
            alg_file = os.path.join(file_output_folder, f"{algorithm}_result.json")
            with open(alg_file, 'w') as f:
                json.dump(result_data, f, indent=4)
            
            # Lưu vào global results
            global_results[algorithm][filename] = result_data
            
            print(f"✅ {algorithm} hoàn thành: {cycle-1} cycles, {total_energy_consumed:.2f}J, {final_alive}/{total_nodes} nodes sống")

        # ======== VẼ BIỂU ĐỒ CHO FILE NÀY ========
        print(f"\n📊 Vẽ biểu đồ so sánh cho {filename}")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Biểu đồ 1: Network Lifetime (cycles)
        ax = axes[0, 0]
        cycles_data = [file_results[alg]['cycles_completed'] for alg in algorithms]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax.bar(algorithms, cycles_data, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Số chu kỳ (cycles)', fontweight='bold', fontsize=12)
        ax.set_title(f'Thời gian sống mạng - {filename}', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Thêm giá trị lên đầu cột
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Biểu đồ 2: Total Energy Consumed
        ax = axes[0, 1]
        energy_data = [file_results[alg]['total_energy_consumed'] for alg in algorithms]
        bars = ax.bar(algorithms, energy_data, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Năng lượng tiêu thụ (J)', fontweight='bold', fontsize=12)
        ax.set_title(f'Tổng năng lượng tiêu thụ - {filename}', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Biểu đồ 3: Số node sống theo chu kỳ
        ax = axes[1, 0]
        for alg, color in zip(algorithms, colors):
            alive_log = file_results[alg]['alive_log']
            ax.plot(range(len(alive_log)), alive_log, marker='o', label=alg, 
                   linewidth=2.5, markersize=4, color=color)
        
        ax.set_xlabel('Số chu kỳ', fontweight='bold', fontsize=12)
        ax.set_ylabel('Số node sống', fontweight='bold', fontsize=12)
        ax.set_title(f'Số node sống theo chu kỳ - {filename}', fontweight='bold', fontsize=14)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=total_nodes*0.1, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Ngưỡng 10%')
        
        # Biểu đồ 4: Tỷ lệ node sống cuối cùng
        ax = axes[1, 1]
        ratio_data = [file_results[alg]['final_alive_ratio'] * 100 for alg in algorithms]
        bars = ax.bar(algorithms, ratio_data, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Tỷ lệ node sống (%)', fontweight='bold', fontsize=12)
        ax.set_title(f'Tỷ lệ node sống cuối cùng - {filename}', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 100])
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        chart_file = os.path.join(file_output_folder, f"comparison_{base_name}.png")
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Biểu đồ đã lưu: {chart_file}")

    # ======== VẼ BIỂU ĐỒ TỔNG HỢP TẤT CẢ CÁC FILE ========
    print(f"\n{'='*80}")
    print("📊 VẼ BIỂU ĐỒ TỔNG HỢP")
    print(f"{'='*80}")
    
    # Chuẩn bị dữ liệu
    node_counts = []
    lifetimes = {alg: [] for alg in algorithms}
    energies = {alg: [] for alg in algorithms}
    alive_logs_550 = {alg: [] for alg in algorithms}
    
    for filename in sorted(files):
        # Trích xuất số node từ tên file
        try:
            num_nodes = int(filename.split('_')[1].split('.')[0])
            node_counts.append(num_nodes)
        except:
            node_counts.append(0)
        
        for alg in algorithms:
            if filename in global_results[alg]:
                lifetimes[alg].append(global_results[alg][filename]['cycles_completed'])
                energies[alg].append(global_results[alg][filename]['total_energy_consumed'])
                
                # Lưu alive_log cho file 550 nodes
                if '550' in filename:
                    alive_logs_550[alg] = global_results[alg][filename]['alive_log']
            else:
                lifetimes[alg].append(0)
                energies[alg].append(0)
    
    # Vẽ 4 biểu đồ tổng hợp
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    x = np.arange(len(node_counts))
    width = 0.25
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Biểu đồ 1: Thời gian sống toàn mạng
    ax = axes[0, 0]
    for i, alg in enumerate(algorithms):
        bars = ax.bar(x + i*width, lifetimes[alg], width, label=alg, 
                     color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
    ax.set_xlabel('Số lượng node', fontweight='bold', fontsize=13)
    ax.set_ylabel('Thời gian sống (cycles)', fontweight='bold', fontsize=13)
    ax.set_title('Thời gian sống toàn mạng theo số node', fontweight='bold', fontsize=15)
    ax.set_xticks(x + width)
    ax.set_xticklabels(node_counts, fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Biểu đồ 2: Năng lượng tiêu thụ
    ax = axes[0, 1]
    for i, alg in enumerate(algorithms):
        bars = ax.bar(x + i*width, energies[alg], width, label=alg, 
                     color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
    ax.set_xlabel('Số lượng node', fontweight='bold', fontsize=13)
    ax.set_ylabel('Năng lượng tiêu thụ (J)', fontweight='bold', fontsize=13)
    ax.set_title('Tổng năng lượng tiêu thụ theo số node', fontweight='bold', fontsize=15)
    ax.set_xticks(x + width)
    ax.set_xticklabels(node_counts, fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Biểu đồ 3: Số node sống theo chu kỳ (550 nodes)
    ax = axes[1, 0]
    for alg, color in zip(algorithms, colors):
        if alive_logs_550[alg]:
            ax.plot(range(len(alive_logs_550[alg])), alive_logs_550[alg], 
                   marker='o', label=alg, linewidth=2.5, markersize=5, color=color)
    
    ax.set_xlabel('Số chu kỳ', fontweight='bold', fontsize=13)
    ax.set_ylabel('Số node sống', fontweight='bold', fontsize=13)
    ax.set_title('Số node sống theo chu kỳ (550 nodes)', fontweight='bold', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Biểu đồ 4: Số chu kỳ theo số lượng nodes
    ax = axes[1, 1]
    for i, alg in enumerate(algorithms):
        ax.plot(node_counts, lifetimes[alg], marker='o', label=alg, 
               linewidth=2.5, markersize=8, color=colors[i])
    
    ax.set_xlabel('Số lượng node', fontweight='bold', fontsize=13)
    ax.set_ylabel('Số chu kỳ', fontweight='bold', fontsize=13)
    ax.set_title('Số chu kỳ mạng theo số lượng node', fontweight='bold', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_chart = os.path.join(output_main_folder, 'summary_all_datasets.png')
    plt.savefig(summary_chart, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Biểu đồ tổng hợp đã lưu: {summary_chart}")
    
    # Lưu kết quả tổng hợp
    summary_data = {
        'node_counts': node_counts,
        'algorithms': algorithms,
        'network_lifetime': lifetimes,
        'total_energy_consumed': energies,
        'detailed_results': global_results
    }
    
    summary_json = os.path.join(output_main_folder, 'summary_all_results.json')
    with open(summary_json, 'w') as f:
        json.dump(summary_data, f, indent=4)
    
    print(f"✅ Kết quả tổng hợp đã lưu: {summary_json}")
    
    # In bảng kết quả
    print(f"\n{'='*100}")
    print("BẢNG KẾT QUẢ SO SÁNH CHI TIẾT")
    print(f"{'='*100}")
    print(f"{'Nodes':<10} {'Greedy':<30} {'GA':<30} {'PSO':<30}")
    print(f"{'':<10} {'Cycles':<10} {'Energy(J)':<10} {'Alive%':<10} {'Cycles':<10} {'Energy(J)':<10} {'Alive%':<10} {'Cycles':<10} {'Energy(J)':<10} {'Alive%':<10}")
    print("-" * 100)
    
    for i, nc in enumerate(node_counts):
        row = f"{nc:<10}"
        fname = sorted(files)[i]
        for alg in algorithms:
            if fname in global_results[alg]:
                cycles = global_results[alg][fname]['cycles_completed']
                energy = global_results[alg][fname]['total_energy_consumed']
                alive_ratio = global_results[alg][fname]['final_alive_ratio'] * 100
                row += f"{cycles:<10} {energy:<10.2f} {alive_ratio:<10.1f}"
            else:
                row += f"{'N/A':<10} {'N/A':<10} {'N/A':<10}"
        print(row)
    
    print(f"\n{'='*100}")
    print(f"✅ HOÀN THÀNH! Tất cả kết quả đã được lưu tại: {output_main_folder}")
    print(f"{'='*100}")

print("✓ Complete main comparison function with detailed folders loaded")
if __name__ == '__main__':
    main()