import math
from numba import njit

class Computing:
    """
    Class chứa các hàm tính toán liên quan đến vận tốc và thời gian di chuyển của AUV
    """
    
    @staticmethod
    @njit
    from numba import njit
import math

class Computing:
    @staticmethod
    @njit
    def compute_vs(p1, p2, v_f, v_AUV):
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1

        L = math.sqrt(dx*dx + dy*dy + dz*dz)
        if L == 0.0:
            return v_AUV

        # Heave velocity v_f is along +z (as in the paper)
        # beta = angle(L, +z) => cos(beta) = (L · z_hat) / |L| = dz / L
        cosb = dz / L
        if cosb > 1.0:
            cosb = 1.0
        elif cosb < -1.0:
            cosb = -1.0

        # v_s = v_f*cos(beta) + sqrt(v_AUV^2 - v_f^2 * sin^2(beta))
        # sin^2(beta) = 1 - cos^2(beta)
        sin2 = 1.0 - cosb*cosb
        sqrt_term = v_AUV*v_AUV - v_f*v_f * sin2

        # Infeasible to keep resultant velocity exactly along L -> avoid NaN
        if sqrt_term < 0.0:
            sqrt_term = 0.0

        v_s = v_f*cosb + math.sqrt(sqrt_term)

        # avoid zero/negative speed for time computation
        if v_s < 1e-9:
            v_s = 1e-9

        return v_s
    
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
    
    def energy_member(best_time, d):
        G, L = 100, 1024
        P_r, P_idle = 0.8e-3, 0.1e-3
        DR = 4000
        E_ELEC = 50e-9      # J/bit
        EPS_FS = 10e-12     # J/bit/m^2

        # Time to transmit
        T_tx = G * L / DR

        # Energy terms (THEO ĐÚNG CÔNG THỨC TRONG ẢNH)
        E_tx = G * L * E_ELEC + G * L * EPS_FS * (d ** 2)
        E_rx = P_r * T_tx
        E_idle = (best_time - 2* T_tx) * P_idle
        E_total = E_tx + E_rx + E_idle

        return E_total, E_tx, E_rx
    
    def energy_cluster_head(best_time, n_members):
        G, L = 100, 1024
        P_t, P_idle = 1.6e-3, 0.1e-3
        DR, DR_i = 4000, 10000
        E_ELEC = 50e-9      # J/bit
        EPS_FS = 10e-12     # J/bit/m^2
        E_AGG = 5e-9  # J/bit

        # Time
        T_rx = G * L * n_members / DR
        T_tx = G * L * n_members / DR_i

        # Energy terms (THEO ĐÚNG CÔNG THỨC TRONG ẢNH)
        E_rx = G * L * n_members * E_ELEC + G * L * n_members * E_AGG
        E_tx = P_t * T_tx
        E_idle = (best_time - T_rx - T_tx) * P_idle
    
        return E_rx + E_tx + E_idle

    def update_energy(all_nodes, node_positions, clusters, best_time):
        
        for cid in clusters:
            cluster = clusters[cid]
            ch = cluster['cluster_head']
            nodes = cluster['nodes']

            # Nếu CH đã chết thì bỏ qua cụm
            if ch not in all_nodes:
                continue

            ch_pos = node_positions[ch]

            # MEMBER NODES
            n_members = 0

            for nid in nodes:
                if nid == ch:
                    continue
                if nid not in all_nodes:
                    continue

                d = math.dist(node_positions[nid], ch_pos)
                E_total, E_tx, E_rx = energy_member(best_time, d)
                all_nodes[nid]['residual_energy'] -= E_total
                if all_nodes[nid]['residual_energy'] < (E_tx + E_rx):
                    all_nodes[nid]['residual_energy'] = 0.0
                    continue
                all_nodes[nid]['residual_energy'] = max(
                    all_nodes[nid]['residual_energy'], 0.0
                )

                n_members += 1

             # CLUSTER HEAD
            E_ch = energy_cluster_head(best_time, n_members)

            all_nodes[ch]['residual_energy'] -= E_ch
            if all_nodes[ch]['residual_energy'] < (E_tx + E_rx):
                all_nodes[ch]['residual_energy'] = 0.0
                continue
            all_nodes[ch]['residual_energy'] = max(
                all_nodes[ch]['residual_energy'], 0.0
            )


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
    
    