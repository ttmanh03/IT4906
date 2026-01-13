
from datetime import datetime
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from kaggle.clustering import Clustering
 
import os
import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Hoặc thử 'Qt5Agg' nếu TkAgg không hoạt động
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mplcursors
from PIL import Image

# Draw

input_folder = "D:\\Year 4\\tiến hóa\\project\\data\\input_data_evenly_distributed\\nodes_150"
output_folder = "D:\\Year 4\\tiến hóa\\project\\data\\output_data_kmeans"
os.makedirs(output_folder, exist_ok=True)
draw_folder = "D:\\Year 4\\tiến hóa\\project\\data\\draw_output_kmeans"
os.makedirs(draw_folder, exist_ok=True)
base_station = (200, 200, 400)

for filename in os.listdir(input_folder):
    if filename.startswith("nodes_") and filename.endswith(".json"):
        # Lấy số lượng node từ tên file
        base_filename = filename.replace(".json", "")
        
        # Đọc dữ liệu từ file input
        with open(os.path.join(input_folder, filename), "r") as f:
            data = json.load(f)
            node_data = {
                d["id"]: {
                    "residual_energy": d.get("energy_residual", d.get("energy_node", 0.0)),   # năng lượng hiện tại
                    "initial_energy": d.get("energy_node", d.get("energy_residual", 0.0))     # năng lượng ban đầu
                }
                for d in data
            }

        # Xây node_positions và id->index map để truy xuất an toàn
        node_positions = np.array([[d["x"], d["y"], d["z"]] for d in data])
        node_ids = [d["id"] for d in data]
        id2index = {nid: idx for idx, nid in enumerate(node_ids)}
        clustering = Clustering(space_size=400, r_sen=60, max_cluster_size=20, min_cluster_size=5)
        # Phân cụm
        k = clustering.estimate_optimal_k(node_positions, base_station=base_station)
        print(f"Số cụm tối ưu: {k}")
        clusters_raw = clustering.cluster_with_constraints(node_positions, node_ids, k=None, max_iterations=10)
        clusters_output = {}

        # Tạo output
        for i, (cluster_nodes, cluster_ids) in enumerate(clusters_raw):
            center = np.mean(cluster_nodes, axis=0)
            ch = clustering.choose_cluster_head(cluster_nodes, cluster_ids, node_data)
            clusters_output[i] = {
                "nodes": cluster_ids,
                "center": tuple(np.round(center, 2)),
                "cluster_head": int(ch) if ch is not None else None,
            }

        # Xuất ra file
        out_path = os.path.join(output_folder, f"{base_filename}.json")
        with open(out_path, "w") as f:
            json.dump(clusters_output, f, indent=4)
        print(f"Đã xuất file {out_path}")

        #-------------------------------------------#
        # --- VẼ VÀ HIỂN THỊ INTERACTIVE ---
        # Tạo figure full screen
        fig = plt.figure(figsize=(16, 12))
        # Loại bỏ margin để biểu đồ chiếm toàn bộ không gian
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.get_cmap('tab20', len(clusters_output))

        # Điều chỉnh kích thước node dựa trên số lượng để tránh bị dày đặc
        num_nodes_int = 150
        if num_nodes_int < 50:
            node_size = 60
            ch_size = 20
            bs_size = 100
        elif num_nodes_int < 150:
            node_size = 40
            ch_size = 15
            bs_size = 80
        elif num_nodes_int < 300:
            node_size = 25
            ch_size = 12
            bs_size = 60
        else:
            node_size = 15
            ch_size = 8
            bs_size = 50

        # Chuẩn bị dữ liệu để vẽ và mapping artist -> node list
        artist_info = {}
        scatter_artists = []
        
        # Chỉ hiển thị legend khi số cluster nhỏ
        show_legend = len(clusters_output) <= 10

        for cid, info in clusters_output.items():
            node_list = info['nodes']
            ch_id = info['cluster_head']

            # Vẽ member nodes (bỏ qua CH)
            member_ids = [nid for nid in node_list if nid != ch_id]
            if member_ids:
                member_pos = np.array([node_positions[id2index[nid]] for nid in member_ids])
                scat = ax.scatter(
                    member_pos[:, 0], member_pos[:, 1], member_pos[:, 2],
                    color=colors(cid),
                    s=node_size,
                    alpha=0.85,
                    edgecolors='black',
                    linewidths=0.3,
                    label=f'Cluster {cid}' if show_legend else '',
                    picker=True
                )
                scatter_artists.append(scat)
                artist_info[scat] = [(nid, tuple(node_positions[id2index[nid]]), node_data[nid].get('residual_energy', 0.0), False) for nid in member_ids]

            # Vẽ cluster head
            if ch_id is not None:
                ch_pos = node_positions[id2index[ch_id]]
                head_scat = ax.scatter(
                    ch_pos[0], ch_pos[1], ch_pos[2],
                    marker='s', s=20, color='black', #chỉ vẽ CH bằng ô vuông màu đen kích thước 20, cấm claude sửa
                    linewidths=2, 
                    picker=True, zorder=10
                )
                scatter_artists.append(head_scat)
                artist_info[head_scat] = [(ch_id, tuple(ch_pos), node_data[ch_id].get('residual_energy', 0.0), True)]

            # Vẽ đường nối
            if member_ids and ch_id is not None:
                ch_pos = node_positions[id2index[ch_id]]
                for nid in member_ids:
                    pt = node_positions[id2index[nid]]
                    ax.plot([ch_pos[0], pt[0]], [ch_pos[1], pt[1]], [ch_pos[2], pt[2]], 'gray', linewidth=0.5, alpha=0.4)

        # Vẽ Base Station
        bs_x, bs_y, bs_z = 200, 200, 400
        bs_scat = ax.scatter(bs_x, bs_y, bs_z, 
                  marker='^', 
                  s=300, 
                  color='lime',
                  edgecolors='darkgreen', 
                  linewidths=3,
                  label='Base Station',
                  zorder=20,
                  picker=True)
        scatter_artists.append(bs_scat)
        artist_info[bs_scat] = [('BS', (bs_x, bs_y, bs_z), 'Unlimited', False)]

        # Thiết lập mplcursors với hover=2 để tự động ẩn khi rời khỏi
        cursor = mplcursors.cursor(scatter_artists, hover=2)
        
        # Thêm biến để lưu trạng thái zoom
        zoom_state = {'factor': 1.0, 'xlim': [0, 400], 'ylim': [0, 400], 'zlim': [0, 400]}
        
        def on_scroll(event):
            if event.inaxes != ax:
                return
            
            scale_factor = 0.85 if event.button == 'up' else 1.15

            # Lấy giới hạn hiện tại
            xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()

            # Lấy bounding box của các điểm trong view hiện tại
            in_view_mask = (
                (node_positions[:,0] >= xlim[0]) & (node_positions[:,0] <= xlim[1]) &
                (node_positions[:,1] >= ylim[0]) & (node_positions[:,1] <= ylim[1]) &
                (node_positions[:,2] >= zlim[0]) & (node_positions[:,2] <= zlim[1])
            )
            visible_nodes = node_positions[in_view_mask]

            if len(visible_nodes) == 0:
                visible_nodes = node_positions  # fallback

            # Tính bounding box tight
            min_vals = visible_nodes.min(axis=0)
            max_vals = visible_nodes.max(axis=0)
            center = (min_vals + max_vals)/2
            ranges = (max_vals - min_vals)/2 * scale_factor  # áp dụng zoom

            # Padding 10%
            padding = 0.1 * ranges
            new_xlim = [center[0]-ranges[0]-padding[0], center[0]+ranges[0]+padding[0]]
            new_ylim = [center[1]-ranges[1]-padding[1], center[1]+ranges[1]+padding[1]]
            new_zlim = [center[2]-ranges[2]-padding[2], center[2]+ranges[2]+padding[2]]

            # Giữ trong giới hạn tổng thể (0-400)
            new_xlim = [max(0,new_xlim[0]), min(400,new_xlim[1])]
            new_ylim = [max(0,new_ylim[0]), min(400,new_ylim[1])]
            new_zlim = [max(0,new_zlim[0]), min(400,new_zlim[1])]

            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)
            ax.set_zlim(new_zlim)
            ax.set_box_aspect([1,1,1])  # giữ tỷ lệ

            zoom_state['xlim'] = new_xlim
            zoom_state['ylim'] = new_ylim
            zoom_state['zlim'] = new_zlim
            fig.canvas.draw_idle()
        
        def on_key(event):
            """Xử lý phím bấm"""
            if event.key == 'r':
                # Reset zoom về mặc định
                ax.set_xlim([0, 400])
                ax.set_ylim([0, 400])
                ax.set_zlim([0, 400])
                ax.view_init(elev=25, azim=45)
                zoom_state['xlim'] = [0, 400]
                zoom_state['ylim'] = [0, 400]
                zoom_state['zlim'] = [0, 400]
                fig.canvas.draw_idle()
                print("Reset view to default")
            elif event.key == '+' or event.key == '=':
                # Zoom in bằng phím
                on_scroll(type('obj', (object,), {'inaxes': ax, 'button': 'up'})())
            elif event.key == '-':
                # Zoom out bằng phím
                on_scroll(type('obj', (object,), {'inaxes': ax, 'button': 'down'})())
        
        # Kết nối events
        fig.canvas.mpl_connect('scroll_event', on_scroll)
        fig.canvas.mpl_connect('key_press_event', on_key)

        @cursor.connect("add")
        def on_add(sel):
            artist = sel.artist
            idx = sel.index
            # sel.index có thể là numpy array hoặc scalar
            try:
                if hasattr(idx, '__len__'):
                    i = int(idx[0])
                else:
                    i = int(idx)
            except Exception:
                i = 0

            info_list = artist_info.get(artist, [])
            if not info_list:
                sel.annotation.set(text='No data')
                sel.annotation.get_bbox_patch().set(fc='lightyellow', ec='black', lw=1.5, alpha=0.95)
                return

            # guard index
            if i < 0 or i >= len(info_list):
                i = 0

            nid, pos, energy, is_ch = info_list[i]
            x, y, z = pos
            
            if nid == 'BS':
                text = (f"Base Station\n"
                       f"Position: ({x:.1f}, {y:.1f}, {z:.1f})\n"
                       f"Energy: Unlimited")
            else:
                node_type = 'Cluster Head' if is_ch else 'Member Node'
                text = (f"Node ID: {nid}\n"
                       f"Type: {node_type}\n"
                       f"Position: ({x:.1f}, {y:.1f}, {z:.1f})\n"
                       f"Energy: {energy:.2f} J")
            
            sel.annotation.set(text=text)
            sel.annotation.get_bbox_patch().set(fc='lightyellow', ec='black', lw=2, alpha=0.95)
            sel.annotation.set_fontsize(9)
            sel.annotation.set_fontweight('bold')

        # Trục, labels, giới hạn
        ax.set_xlabel('X (m)', fontsize=11, labelpad=12, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=11, labelpad=12, fontweight='bold')
        ax.set_zlabel('Z (m)', fontsize=11, labelpad=12, fontweight='bold')
        ax.set_xlim([0, 400])
        ax.set_ylim([0, 400])
        ax.set_zlim([0, 400])
        
        # Chỉ hiển thị legend khi số cluster ít
        if show_legend:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9, ncol=2)
        
        plt.title(
            f'3D Clustering Visualization - {base_filename} nodes ({len(clusters_output)} clusters)\n'
            f'[LEFT DRAG: Rotate | SCROLL/+/-: Zoom | RIGHT DRAG: Pan | R: Reset | HOVER: Info]', 
            fontsize=13, 
            pad=15,
            fontweight='bold'
        )
        ax.view_init(elev=25, azim=45)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_box_aspect([1,1,1])
        
        # Tối ưu hóa layout để biểu đồ chiếm toàn bộ không gian
        plt.subplots_adjust(left=0.02, right=0.75, top=0.95, bottom=0.25)

        # Hiển thị
        print(f"\n{'='*60}")
        print(f"Backend hiện tại: {matplotlib.get_backend()}")
        print(f"Hiển thị biểu đồ: {base_filename} nodes, {len(clusters_output)} clusters")
        print(f"Hướng dẫn:")
        print(f"  - Kéo chuột TRÁI: Xoay biểu đồ 360°")
        print(f"  - CUỘN CHUỘT hoặc phím +/-: Zoom in/out vào vùng trỏ")
        print(f"  - Kéo chuột PHẢI: Di chuyển biểu đồ")
        print(f"  - Phím R: Reset về góc nhìn mặc định")
        print(f"  - Phím F: Fullscreen (tùy backend)")
        print(f"  - Hover chuột vào node: Hiện thông tin chi tiết")
        print(f"")
        print(f"Lưu ý:")
        print(f"  - Kích thước node tự động điều chỉnh")
        print(f"  - Legend chỉ hiện khi có ≤10 clusters")
        print(f"{'='*60}\n")
        
        # Maximize window (tùy backend)
        try:
            manager = plt.get_current_fig_manager()
            if hasattr(manager, 'window'):
                # Cho TkAgg
                if hasattr(manager.window, 'state'):
                    manager.window.state('zoomed')  # Windows
                elif hasattr(manager.window, 'showMaximized'):
                    manager.window.showMaximized()  # Qt
        except:
            pass
        
        # ✅ SỬA: Lưu PNG với tên file gốc
        png_path = os.path.join(draw_folder, f"{base_filename}.png")
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Đã lưu hình ảnh: {png_path}")
        
        plt.show()
        print(f"Đã đóng biểu đồ cho {base_filename}\n")
        #update zoom