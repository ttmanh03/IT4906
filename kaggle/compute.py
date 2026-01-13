import math
from numba import njit

class Computing:
    """
    Class chứa các hàm tính toán liên quan đến vận tốc và thời gian di chuyển của AUV
    """
    
    @staticmethod
    @njit
    def compute_vs(p1, p2, v_f, v_AUV):
        """
        Tính vận tốc thực tế của AUV khi di chuyển từ p1 đến p2
        
        Args:
            p1: tuple (x1, y1, z1) - tọa độ điểm xuất phát
            p2: tuple (x2, y2, z2) - tọa độ điểm đích
            v_f: float - vận tốc dòng chảy
            v_AUV: float - vận tốc của AUV
            
        Returns:
            float - vận tốc thực tế
        """
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        Lx = x2 - x1
        Ly = y2 - y1
        Lz = z2 - z1

        L_mag = math.sqrt(Lx*Lx + Ly*Ly + Lz*Lz)
        if L_mag == 0.0:
            return v_AUV

        # cos(beta)
        cos_beta = Lz / L_mag
        if cos_beta > 1.0:
            cos_beta = 1.0
        elif cos_beta < -1.0:
            cos_beta = -1.0

        beta = math.acos(cos_beta)

        inner = (v_f * cos_beta) / v_AUV
        if inner > 1.0:
            inner = 1.0
        elif inner < -1.0:
            inner = -1.0

        angle = beta + math.acos(inner)

        if abs(cos_beta) < 1e-9:
            return v_AUV

        return abs(math.cos(angle) * v_AUV / cos_beta)
    
    @staticmethod
    @njit
    def travel_time(path, coords, v_f, v_AUV):
        """
        Tính tổng thời gian di chuyển theo đường đi cho trước
        
        Args:
            path: list/array - danh sách các index của các điểm theo thứ tự
            coords: array - mảng tọa độ các điểm (n x 3)
            v_f: float - vận tốc dòng chảy
            v_AUV: float - vận tốc của AUV
            
        Returns:
            float - tổng thời gian di chuyển (bao gồm quay về điểm xuất phát)
        """
        total_time = 0.0
        n = len(path)
        if n <= 1:
            return 0.0

        # Các cạnh trong chu trình
        for i in range(n - 1):
            i1 = path[i]
            i2 = path[i+1]

            p1 = coords[i1]
            p2 = coords[i2]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dz = p2[2] - p1[2]
            d = math.sqrt(dx*dx + dy*dy + dz*dz)

            v_s = Computing.compute_vs(p1, p2, v_f, v_AUV)
            if v_s < 1e-9:
                v_s = 1e-9

            total_time += d / v_s

        # Quay về điểm bắt đầu
        p1 = coords[path[-1]]
        p2 = coords[path[0]]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        d = math.sqrt(dx*dx + dy*dy + dz*dz)

        v_s = Computing.compute_vs(p1, p2, v_f, v_AUV)
        if v_s < 1e-9:
            v_s = 1e-9

        total_time += d / v_s

        return total_time
    
    def compute_energy(best_time, n_members):
        """
        Tính năng lượng tiêu thụ cho Member Node và Cluster Head.
        
        Parameters:
        - best_time: Thời gian hoàn thành chu kỳ AUV
        - n_members: Số lượng node thành viên thực tế trong cluster (không tính cluster head)
        """
        G, L = 100, 1024
        P_t, P_r, P_idle, DR, DR_i = 1.6e-3, 0.8e-3, 0.1e-3, 4000, 10000

        # Năng lượng cho Member Node
        E_tx_MN = G * P_t * L / DR
        E_rx_MN = G * P_r * L / DR
        E_idle_MN = (best_time - G * L / DR - G * P_r * L / DR) * P_idle
        E_total_MN = E_tx_MN + E_idle_MN + E_rx_MN

        # Năng lượng cho Cluster Head (nhận từ n_members node, truyền cho AUV)
        E_rx_TN = G * P_r * L * n_members / DR
        E_tx_TN = G * P_t * L * n_members / DR_i
        #E_idle_TN = (best_time - (G*L*n_members/DR) - (G*L*n_members/DR_i)) * P_idle
        E_idle_TN = (best_time - (G*L*n_members/DR)) * P_idle
        E_total_TN = E_rx_TN + E_tx_TN + E_idle_TN

        return {
            "Member": {"E_total": E_total_MN},
            "Target": {"E_total": E_total_TN}
        }

    def update_energy(all_nodes, clusters, best_time):
        """
        Cập nhật năng lượng cho tất cả các node dựa trên số member thực tế của từng cluster.
        
        Parameters:
        - all_nodes: Dictionary chứa thông tin tất cả các node
        - clusters: Dictionary chứa thông tin các cluster
        - best_time: Thời gian hoàn thành chu kỳ AUV
        """
        for cid, cinfo in clusters.items():
            ch = cinfo.get('cluster_head')
            nodes = cinfo.get('nodes', [])
            
            # Tính số member nodes (không tính cluster head)
            n_members = len([n for n in nodes if n != ch])
            
            # Tính năng lượng cho cluster này với số member thực tế
            energy_report = Computing.compute_energy(best_time, n_members)
            
            for nid in nodes:
                if nid not in all_nodes: continue
                if nid == ch:
                    all_nodes[nid]['residual_energy'] -= energy_report['Target']['E_total']
                else:
                    all_nodes[nid]['residual_energy'] -= energy_report['Member']['E_total']
                all_nodes[nid]['residual_energy'] = max(all_nodes[nid]['residual_energy'], 0.0)

    def remove_dead_nodes(all_nodes, clusters):
        """
        Loại bỏ các node đã hết năng lượng và cập nhật lại clusters.
        
        Returns:
        - new_clusters: Dictionary các cluster còn node sống
        - dead: List các node_id đã chết
        """
        dead = [nid for nid, info in list(all_nodes.items()) if info['residual_energy'] <= 0]
        for nid in dead:
            del all_nodes[nid]

        new_clusters = {}
        for cid, cinfo in clusters.items():
            alive_nodes = [nid for nid in cinfo.get('nodes', []) if nid in all_nodes]
            if alive_nodes:
                new_c = dict(cinfo)
                new_c['nodes'] = alive_nodes
                new_clusters[cid] = new_c

        return new_clusters, dead
    
    