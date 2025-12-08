import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_result_file(result_path):
    """Đọc file kết quả JSON"""
    with open(result_path, 'r') as f:
        return json.load(f)

def load_node_positions(input_folder, input_file):
    """Đọc vị trí các node từ file input"""
    input_path = os.path.join(input_folder, input_file)
    if not os.path.exists(input_path):
        # Thử tìm trong thư mục input_data
        input_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input_data", input_file)
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    positions = {}
    for node in data:
        positions[node['id']] = np.array([node['x'], node['y'], node['z']])
    
    return positions

def visualize_death_events(result_path, output_path=None):
    """
    Vẽ mô phỏng 3D các node, tô màu theo sự kiện chết
    
    Parameters:
    - result_path: Đường dẫn đến file kết quả JSON (ví dụ: result_nodes_150.json)
    - output_path: Đường dẫn lưu hình (nếu None sẽ tự động tạo)
    """
    # Đọc file kết quả
    result = load_result_file(result_path)
    
    # Lấy thông tin file input
    input_file = result.get('input_file', 'nodes_150.json')
    
    # Tìm thư mục chứa file input
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.dirname(os.path.abspath(result_path))
    
    possible_input_folders = [
        os.path.join(script_dir, "..", "input_data"),  # IT4906/input_data
        os.path.join(script_dir, "input_data"),         # IT4906/kaggle/input_data
        os.path.join(result_dir, "..", "..", "..", "input_data"),  # Từ output/ga/ lên input_data
        "/kaggle/input/nodes-data",
    ]
    
    node_positions = None
    for folder in possible_input_folders:
        folder = os.path.normpath(folder)
        input_path = os.path.join(folder, input_file)
        if os.path.exists(input_path):
            try:
                node_positions = load_node_positions(folder, input_file)
                print(f"✅ Tìm thấy file input tại: {input_path}")
                break
            except Exception as e:
                print(f"⚠️ Lỗi đọc file {input_path}: {e}")
                continue
    
    if node_positions is None:
        print(f"❌ Không tìm thấy file input: {input_file}")
        print(f"   Đã tìm trong các thư mục:")
        for folder in possible_input_folders:
            print(f"   - {os.path.normpath(folder)}")
        return
    
    # Lấy thông tin death events
    death_events = result.get('death_events', [])
    total_nodes = result.get('nodes_summary', {}).get('initial_total_nodes', len(node_positions))
    
    # Tạo mapping: node_id -> death_event_index (None nếu còn sống)
    node_death_event = {}
    for event_idx, event in enumerate(death_events):
        for node_id in event.get('dead_ids', []):
            node_death_event[node_id] = event_idx
    
    # Tạo figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Tạo colormap cho các death events
    num_events = len(death_events)
    if num_events > 0:
        cmap = plt.colormaps.get_cmap('tab10' if num_events <= 10 else 'tab20')
    
    # Vẽ các node còn sống (màu xám)
    alive_nodes = []
    alive_positions = []
    for node_id, pos in node_positions.items():
        if node_id not in node_death_event:
            alive_nodes.append(node_id)
            alive_positions.append(pos)
    
    if alive_positions:
        alive_positions = np.array(alive_positions)
        ax.scatter(alive_positions[:, 0], alive_positions[:, 1], alive_positions[:, 2],
                  c='gray', alpha=0.5, s=50, label=f'Còn sống ({len(alive_nodes)} nodes)')
    
    # Vẽ các node chết theo từng sự kiện
    for event_idx, event in enumerate(death_events):
        dead_ids = event.get('dead_ids', [])
        cycle = event.get('cycle', 'N/A')
        
        if not dead_ids:
            continue
        
        dead_positions = np.array([node_positions[nid] for nid in dead_ids if nid in node_positions])
        
        if len(dead_positions) > 0:
            color = cmap(event_idx % 20)
            ax.scatter(dead_positions[:, 0], dead_positions[:, 1], dead_positions[:, 2],
                      c=[color], s=100, marker='o', edgecolor='black', linewidth=1,
                      label=f'Chết cycle {cycle} ({len(dead_ids)} nodes)')
    
    # Vẽ base station
    base_station = np.array([200, 200, 400])
    ax.scatter(base_station[0], base_station[1], base_station[2],
              c='red', marker='^', s=400, edgecolor='black', linewidth=2,
              label='Base Station', zorder=100)
    
    # Cấu hình axes
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    
    # Tiêu đề
    method = result.get('method', 'Unknown')
    title = f"Death Events Visualization - {method}\n"
    title += f"Total: {total_nodes} nodes | Dead: {len(node_death_event)} | Alive: {total_nodes - len(node_death_event)}"
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    
    # View angle
    ax.view_init(elev=25, azim=45)
    
    # Tight layout
    plt.tight_layout()
    
    # Lưu hình
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(result_path))[0]
        output_folder = os.path.join(script_dir, "death_visualizations")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"{base_name}_death_events.png")
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Đã lưu hình: {output_path}")
    
    # Hiển thị thông tin death events
    print(f"\n📊 Thống kê Death Events:")
    print(f"   Tổng số node: {total_nodes}")
    print(f"   Số node chết: {len(node_death_event)}")
    print(f"   Số node sống: {total_nodes - len(node_death_event)}")
    print(f"\n   Chi tiết các sự kiện chết:")
    for event_idx, event in enumerate(death_events):
        cycle = event.get('cycle', 'N/A')
        dead_count = event.get('dead_count', len(event.get('dead_ids', [])))
        print(f"   - Sự kiện {event_idx + 1}: Cycle {cycle}, {dead_count} nodes chết")
    
    plt.close(fig)
    return output_path

def main():
    """Main function"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if len(sys.argv) >= 2:
        # Nếu có argument dòng lệnh
        result_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # Nếu không có argument, hỏi người dùng
        result_path = input("Nhập đường dẫn file kết quả JSON (ví dụ: output/ga/result_nodes_150.json): ").strip()
        output_path = None  # Để tự động tạo đường dẫn output
    
    # Chuyển đổi đường dẫn tương đối thành tuyệt đối
    if not os.path.isabs(result_path):
        # Thử tìm file theo thứ tự ưu tiên
        possible_paths = [
            os.path.join(script_dir, result_path),  # Từ thư mục chứa script (kaggle/)
            os.path.abspath(result_path),            # Từ thư mục hiện tại
        ]
        
        result_path = None
        for path in possible_paths:
            if os.path.exists(path):
                result_path = path
                break
        
        if result_path is None:
            print(f"❌ File không tồn tại. Đã tìm trong:")
            for path in possible_paths:
                print(f"   - {path}")
            sys.exit(1)
    
    if not os.path.exists(result_path):
        print(f"❌ File không tồn tại: {result_path}")
        sys.exit(1)
    
    print(f"📂 Đọc file: {result_path}")
    visualize_death_events(result_path, output_path)

if __name__ == '__main__':
    main()
