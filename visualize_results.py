"""
Script để vẽ minh họa và so sánh kết quả mô phỏng mạng cảm biến dưới nước
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_result_files(result_dir):
    """
    Đọc tất cả các file result JSON từ thư mục.
    
    Parameters:
    - result_dir: Đường dẫn đến thư mục chứa kết quả
    
    Returns:
    - Dictionary với key là tên file, value là dữ liệu JSON
    """
    results = {}
    result_files = [f for f in os.listdir(result_dir) if f.startswith('result_') and f.endswith('.json')]
    
    for filename in sorted(result_files):
        filepath = os.path.join(result_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Lấy tên dataset từ input_file
            dataset_name = data.get('input_file', filename.replace('result_', '').replace('.json', ''))
            results[dataset_name] = data
    
    return results


def plot_comparison_bar_chart(results, output_dir=None):
    """
    Vẽ biểu đồ cột so sánh số chu kỳ hoàn thành giữa các dataset.
    
    Parameters:
    - results: Dictionary chứa kết quả từ load_result_files()
    - output_dir: Thư mục lưu hình ảnh (optional)
    """
    datasets = list(results.keys())
    cycles = [results[d]['cycles_completed'] for d in datasets]
    nodes = [results[d]['initial_total_nodes'] for d in datasets]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Tạo vị trí cho các cột
    x = np.arange(len(datasets))
    width = 0.6
    
    # Vẽ cột với màu gradient theo giá trị
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(datasets)))
    bars = ax.bar(x, cycles, width, color=colors, edgecolor='black', linewidth=1.5)
    
    # Thêm giá trị lên đầu mỗi cột
    for i, (bar, cycle, node) in enumerate(zip(bars, cycles, nodes)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{cycle} cycles\n({node} nodes)',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('Số chu kỳ hoàn thành', fontsize=14, fontweight='bold')
    ax.set_title('So sánh hiệu suất mạng theo số lượng nodes', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=0, ha='center')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'comparison_bar_chart.png'), dpi=150, bbox_inches='tight')
        print(f"✅ Đã lưu: comparison_bar_chart.png")
    
    plt.show()
    plt.close()


def plot_metrics_comparison(results, output_dir=None):
    """
    Vẽ biểu đồ so sánh nhiều metrics: cycles, alive ratio, nodes.
    
    Parameters:
    - results: Dictionary chứa kết quả từ load_result_files()
    - output_dir: Thư mục lưu hình ảnh (optional)
    """
    datasets = list(results.keys())
    
    # Thu thập dữ liệu
    initial_nodes = [results[d]['initial_total_nodes'] for d in datasets]
    cycles = [results[d]['cycles_completed'] for d in datasets]
    final_nodes = [results[d]['final_alive_nodes'] for d in datasets]
    alive_ratios = [results[d]['final_alive_ratio'] * 100 for d in datasets]
    
    # Tạo subplot với 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Phân tích hiệu suất mạng cảm biến dưới nước', fontsize=18, fontweight='bold', y=0.995)
    
    # 1. Số chu kỳ hoàn thành
    ax1 = axes[0, 0]
    bars1 = ax1.bar(datasets, cycles, color='steelblue', edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Số chu kỳ', fontsize=12, fontweight='bold')
    ax1.set_title('Số chu kỳ hoàn thành', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, cycles):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Tỷ lệ node sống cuối cùng
    ax2 = axes[0, 1]
    bars2 = ax2.bar(datasets, alive_ratios, color='forestgreen', edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Tỷ lệ (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Tỷ lệ nodes còn sống cuối chu kỳ', fontsize=14, fontweight='bold')
    ax2.axhline(y=90, color='red', linestyle='--', linewidth=2, label='Ngưỡng dừng (90%)')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    for bar, val in zip(bars2, alive_ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. So sánh nodes ban đầu vs cuối
    ax3 = axes[1, 0]
    x = np.arange(len(datasets))
    width = 0.35
    bars3a = ax3.bar(x - width/2, initial_nodes, width, label='Nodes ban đầu', 
                     color='skyblue', edgecolor='black', linewidth=1.5)
    bars3b = ax3.bar(x + width/2, final_nodes, width, label='Nodes còn sống', 
                     color='coral', edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Số lượng nodes', fontsize=12, fontweight='bold')
    ax3.set_title('So sánh số lượng nodes', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Hiệu suất theo tỷ lệ (cycles per node)
    ax4 = axes[1, 1]
    cycles_per_node = [c / n for c, n in zip(cycles, initial_nodes)]
    bars4 = ax4.bar(datasets, cycles_per_node, color='mediumpurple', edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Chu kỳ / Node', fontsize=12, fontweight='bold')
    ax4.set_title('Hiệu suất (Cycles per Node)', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars4, cycles_per_node):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
        print(f"✅ Đã lưu: metrics_comparison.png")
    
    plt.show()
    plt.close()


def plot_summary_table(results, output_dir=None):
    """
    Tạo bảng tổng hợp kết quả dưới dạng hình ảnh.
    
    Parameters:
    - results: Dictionary chứa kết quả từ load_result_files()
    - output_dir: Thư mục lưu hình ảnh (optional)
    """
    datasets = list(results.keys())
    
    # Chuẩn bị dữ liệu cho bảng
    table_data = []
    for dataset in datasets:
        row = [
            dataset,
            results[dataset]['initial_total_nodes'],
            results[dataset]['cycles_completed'],
            results[dataset]['final_alive_nodes'],
            f"{results[dataset]['final_alive_ratio']*100:.1f}%",
            f"{results[dataset]['cycles_completed'] / results[dataset]['initial_total_nodes']:.2f}"
        ]
        table_data.append(row)
    
    # Tạo figure
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Tạo bảng
    headers = ['Dataset', 'Nodes\nban đầu', 'Chu kỳ\nhoàn thành', 'Nodes\ncòn sống', 'Tỷ lệ\nsống (%)', 'Hiệu suất\n(cycles/node)']
    table = ax.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.2])
    
    # Định dạng bảng
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Màu header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Màu xen kẽ cho các hàng
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            else:
                table[(i, j)].set_facecolor('#F2F2F2')
    
    plt.title('Bảng tổng hợp kết quả mô phỏng', fontsize=16, fontweight='bold', pad=20)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'summary_table.png'), dpi=150, bbox_inches='tight')
        print(f"✅ Đã lưu: summary_table.png")
    
    plt.show()
    plt.close()


def plot_scalability_analysis(results, output_dir=None):
    """
    Vẽ biểu đồ phân tích khả năng mở rộng (scalability) của mạng.
    
    Parameters:
    - results: Dictionary chứa kết quả từ load_result_files()
    - output_dir: Thư mục lưu hình ảnh (optional)
    """
    # Sắp xếp theo số nodes tăng dần
    sorted_results = sorted(results.items(), key=lambda x: x[1]['initial_total_nodes'])
    
    nodes = [r[1]['initial_total_nodes'] for r in sorted_results]
    cycles = [r[1]['cycles_completed'] for r in sorted_results]
    labels = [r[0] for r in sorted_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Phân tích khả năng mở rộng (Scalability)', fontsize=16, fontweight='bold')
    
    # 1. Biểu đồ đường: Cycles vs Nodes
    ax1.plot(nodes, cycles, marker='o', linewidth=3, markersize=12, 
             color='steelblue', markeredgecolor='black', markeredgewidth=2)
    ax1.set_xlabel('Số lượng nodes ban đầu', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Số chu kỳ hoàn thành', fontsize=12, fontweight='bold')
    ax1.set_title('Mối quan hệ giữa số nodes và chu kỳ', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Thêm labels cho mỗi điểm
    for x, y, label in zip(nodes, cycles, labels):
        ax1.annotate(f'{y} cycles', xy=(x, y), xytext=(10, 10),
                    textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # 2. Scatter plot với kích thước tương ứng
    sizes = [n * 3 for n in nodes]  # Scale để hiển thị rõ hơn
    scatter = ax2.scatter(nodes, cycles, s=sizes, alpha=0.6, c=cycles, 
                         cmap='viridis', edgecolors='black', linewidths=2)
    ax2.set_xlabel('Số lượng nodes ban đầu', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Số chu kỳ hoàn thành', fontsize=12, fontweight='bold')
    ax2.set_title('Phân tích tương quan (kích thước = số nodes)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Thêm colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Số chu kỳ', fontsize=11, fontweight='bold')
    
    # Thêm labels
    for x, y, label in zip(nodes, cycles, labels):
        ax2.annotate(label, xy=(x, y), xytext=(5, 5),
                    textcoords='offset points', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'scalability_analysis.png'), dpi=150, bbox_inches='tight')
        print(f"✅ Đã lưu: scalability_analysis.png")
    
    plt.show()
    plt.close()


def generate_all_visualizations(result_dir, output_dir=None):
    """
    Tạo tất cả các biểu đồ so sánh và phân tích.
    
    Parameters:
    - result_dir: Đường dẫn đến thư mục chứa file kết quả
    - output_dir: Thư mục lưu hình ảnh (mặc định là result_dir)
    """
    if output_dir is None:
        output_dir = result_dir
    
    print(f"\n{'='*60}")
    print(f"📊 Bắt đầu tạo visualizations từ: {result_dir}")
    print(f"{'='*60}\n")
    
    # Load dữ liệu
    results = load_result_files(result_dir)
    
    if not results:
        print("❌ Không tìm thấy file kết quả nào!")
        return
    
    print(f"✅ Đã tìm thấy {len(results)} file kết quả:")
    for dataset, data in results.items():
        print(f"   - {dataset}: {data['initial_total_nodes']} nodes, {data['cycles_completed']} cycles")
    
    print(f"\n🎨 Đang tạo các biểu đồ...\n")
    
    # Tạo các biểu đồ
    plot_comparison_bar_chart(results, output_dir)
    plot_metrics_comparison(results, output_dir)
    plot_summary_table(results, output_dir)
    plot_scalability_analysis(results, output_dir)
    
    print(f"\n{'='*60}")
    print(f"✅ Hoàn thành! Tất cả biểu đồ đã được lưu tại: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Tìm đường dẫn đúng của thư mục result
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Thử các đường dẫn có thể
    possible_paths = [
        os.path.join(current_dir, "result_ga_ch_most_energy"),
        os.path.join(os.path.dirname(current_dir), "result_ga_ch_most_energy"),
        r"l:\Tính toán tiến hóa\IT4906_Project\result_ga_ch_most_energy",
        r"l:\Tính toán tiến hóa\IT4906_Project\IT4906\result_ga_ch_most_energy"
    ]
    
    result_directory = None
    for path in possible_paths:
        if os.path.exists(path):
            result_directory = path
            print(f"✅ Tìm thấy thư mục kết quả: {path}")
            break
    
    if result_directory is None:
        print("❌ Không tìm thấy thư mục result_ga_ch_most_energy!")
        print("📁 Các đường dẫn đã thử:")
        for path in possible_paths:
            print(f"   - {path}")
        print("\nVui lòng chỉnh sửa đường dẫn trong script hoặc chạy từ đúng thư mục.")
    else:
        generate_all_visualizations(result_directory)
