import sys
import os
import random
import numpy as np
from numba import njit
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from compute import Computing
class Pso_routing:
    """
    Class chứa các thuật toán PSO để tối ưu hóa routing cho TSP 3D
    """
    
    @staticmethod
    def get_swap_sequence(A, B):
        """
        Lấy chuỗi các thao tác swap để biến đổi A thành B
        
        Args:
            A: list - đường đi nguồn
            B: list - đường đi đích
            
        Returns:
            list of tuples - danh sách các cặp (i, j) cần swap
        """
        seq = []
        temp = A.copy()
        for i in range(1, len(A)):  # Bắt đầu từ 1 vì 0 là cố định
            if temp[i] != B[i]:
                # Tìm vị trí của phần tử B[i] trong temp
                try:
                    j = temp.index(B[i])
                    seq.append((i, j))
                    temp[i], temp[j] = temp[j], temp[i]
                except ValueError:
                    # Xảy ra khi A và B không có cùng các phần tử (không nên xảy ra)
                    pass
        return seq
    
    @staticmethod
    def apply_velocity(position, velocity):
        """
        Áp dụng velocity (chuỗi swap) lên position
        
        Args:
            position: list - đường đi hiện tại
            velocity: list of tuples - chuỗi các swap operations
            
        Returns:
            list - đường đi mới sau khi áp dụng velocity
        """
        pos = position.copy()
        for i, j in velocity:
            if i > 0 and j > 0 and i < len(pos) and j < len(pos):  # Bảo vệ index
                pos[i], pos[j] = pos[j], pos[i]
        return pos
    
    @staticmethod
    def pso_tsp_3d_time(coords, v_f=1.0, v_AUV=3.0,
                        n_particles=40, max_iter=200,
                        w=0.5, c1=0.25, c2=0.25, init_gbest=None, verbose=True):
        """
        PSO cơ bản cho TSP 3D với tối ưu hóa thời gian
        
        Args:
            coords: array - tọa độ các điểm (n x 3)
            v_f: float - vận tốc dòng chảy
            v_AUV: float - vận tốc AUV
            n_particles: int - số lượng particles
            max_iter: int - số vòng lặp tối đa
            w: float - trọng số inertia (không dùng trong code này)
            c1: float - hệ số học từ pbest
            c2: float - hệ số học từ gbest
            init_gbest: list - giải pháp tốt nhất ban đầu (optional)
            verbose: bool - hiển thị log
            
        Returns:
            tuple: (gbest, gbest_cost) - đường đi tốt nhất và chi phí
        """
        n_cities = len(coords)
        if n_cities < 2:
            return [0], 0.0
        if n_cities == 2:
            return [0, 1], Computing.travel_time(np.array([0, 1]), coords, v_f, v_AUV)
        
        cities = list(range(1, n_cities))
        # Khởi tạo quần thể
        swarm = [[0] + random.sample(cities, len(cities)) for _ in range(n_particles)]
        velocities = [[] for _ in range(n_particles)]
        
        # Đánh giá ban đầu
        costs = [Computing.travel_time(np.array(p), coords, v_f, v_AUV) for p in swarm]
        pbest = [p.copy() for p in swarm]
        pbest_cost = costs.copy()
        
        # Xác định gbest
        if init_gbest is not None and len(init_gbest) == n_cities:
            gbest = init_gbest.copy()
            gbest_cost = Computing.travel_time(np.array(gbest), coords, v_f, v_AUV)
        else:
            best_idx = int(np.argmin(pbest_cost))
            gbest = pbest[best_idx].copy()
            gbest_cost = pbest_cost[best_idx]
        
        # --- Vòng lặp chính ---
        for t in range(max_iter):
            inertia = 0.7 - 0.5 * (t / max_iter)  # Giảm dần inertia
            
            for i in range(n_particles):
                xi, vi = swarm[i], velocities[i]
                
                # Giữ lại 1 phần vận tốc cũ
                v_new = vi[:int(inertia * len(vi))]
                
                # Ảnh hưởng cá nhân (pbest)
                if random.random() < c1:
                    seq_pb = Pso_routing.get_swap_sequence(xi, pbest[i])
                    if seq_pb:
                        v_new += random.sample(seq_pb, min(2, len(seq_pb)))  # Thêm 2 swap ngẫu nhiên
                
                # Ảnh hưởng toàn cục (gbest)
                if random.random() < c2:
                    seq_gb = Pso_routing.get_swap_sequence(xi, gbest)
                    if seq_gb:
                        v_new += random.sample(seq_gb, min(2, len(seq_gb)))  # Thêm 2 swap ngẫu nhiên
                
                # Cập nhật vị trí và vận tốc
                new_x = Pso_routing.apply_velocity(xi, v_new)
                new_cost = Computing.travel_time(np.array(new_x), coords, v_f, v_AUV)
                
                swarm[i], velocities[i] = new_x, v_new
                
                # Cập nhật pbest và gbest
                if new_cost < pbest_cost[i]:
                    pbest[i], pbest_cost[i] = new_x, new_cost
                    if new_cost < gbest_cost:
                        gbest, gbest_cost = new_x, new_cost
            
            if verbose and t % 50 == 0:
                print(f"    [PSO Iter {t:3d}]: Best time = {gbest_cost:.4f}")
        
        if verbose:
            print(f"    [PSO Iter {max_iter:3d}]: Final Best time = {gbest_cost:.4f}")
        
        return gbest, gbest_cost
    
    @staticmethod
    def pso_tsp_3d_time1(coords, v_f=1.0, v_AUV=3.0,
                         n_particles=40, max_iter=200,
                         w=0.5, c1=0.25, c2=0.25, init_gbest=None, verbose=True):
        """
        PSO với Adaptive Noise - tự động điều chỉnh noise khi bị stuck
        """
        n_cities = len(coords)
        if n_cities < 2:
            return [0], 0.0
        if n_cities == 2:
            return [0, 1], Computing.travel_time(np.array([0, 1]), coords, v_f, v_AUV)
        
        cities = list(range(1, n_cities))
        # Khởi tạo quần thể
        swarm = [[0] + random.sample(cities, len(cities)) for _ in range(n_particles)]
        velocities = [[] for _ in range(n_particles)]
        
        # Đánh giá ban đầu
        costs = [Computing.travel_time(np.array(p), coords, v_f, v_AUV) for p in swarm]
        pbest = [p.copy() for p in swarm]
        pbest_cost = costs.copy()
        
        # Xác định gbest
        if init_gbest is not None and len(init_gbest) == n_cities:
            gbest = init_gbest.copy()
            gbest_cost = Computing.travel_time(np.array(gbest), coords, v_f, v_AUV)
        else:
            best_idx = int(np.argmin(pbest_cost))
            gbest = pbest[best_idx].copy()
            gbest_cost = pbest_cost[best_idx]
        
        # ================= ADAPTIVE NOISE SETUP =================
        no_improve = 0
        last_best = gbest_cost
        p_noise = 0.1
        max_noise_swaps = 2
        # =======================================================
        
        # --- Vòng lặp chính ---
        for t in range(max_iter):
            inertia = 0.7 - 0.5 * (t / max_iter)  # Giữ nguyên công thức
            
            for i in range(n_particles):
                xi, vi = swarm[i], velocities[i]
                
                # Giữ lại 1 phần vận tốc cũ
                v_new = vi[:int(inertia * len(vi))]
                
                # Ảnh hưởng cá nhân (pbest)
                if random.random() < c1:
                    seq_pb = Pso_routing.get_swap_sequence(xi, pbest[i])
                    if seq_pb:
                        v_new += random.sample(seq_pb, min(2, len(seq_pb)))
                
                # Ảnh hưởng toàn cục (gbest)
                if random.random() < c2:
                    seq_gb = Pso_routing.get_swap_sequence(xi, gbest)
                    if seq_gb:
                        v_new += random.sample(seq_gb, min(2, len(seq_gb)))
                
                # ================= ADAPTIVE NOISE =================
                if random.random() < p_noise:
                    k = random.randint(1, max_noise_swaps)
                    for _ in range(k):
                        i1, i2 = random.sample(range(1, n_cities), 2)
                        v_new.append((i1, i2))
                # =================================================
                
                # Cập nhật vị trí
                new_x = Pso_routing.apply_velocity(xi, v_new)
                new_cost = Computing.travel_time(np.array(new_x), coords, v_f, v_AUV)
                
                swarm[i], velocities[i] = new_x, v_new
                
                # Cập nhật pbest và gbest
                if new_cost < pbest_cost[i]:
                    pbest[i], pbest_cost[i] = new_x, new_cost
                    if new_cost < gbest_cost:
                        gbest, gbest_cost = new_x, new_cost
            
            # ============ ADAPTIVE NOISE UPDATE ============
            if gbest_cost >= last_best - 1e-6:
                no_improve += 1
            else:
                no_improve = 0
                last_best = gbest_cost
            
            if no_improve > 15:
                p_noise = min(0.4, p_noise * 1.3)
            else:
                p_noise = max(0.05, p_noise * 0.95)
            # ===============================================
            
            if verbose and t % 50 == 0:
                print(f"    [PSO Iter {t:3d}]: Best time = {gbest_cost:.4f}")
        
        if verbose:
            print(f"    [PSO Iter {max_iter:3d}]: Final Best time = {gbest_cost:.4f}")
        
        return gbest, gbest_cost
    
    @staticmethod
    def pso_tsp_3d_time2(coords, v_f=1.0, v_AUV=3.0,
                         n_particles=40, max_iter=200,
                         w=0.5, c1=0.25, c2=0.25, init_gbest=None, verbose=True):
        """
        PSO với Lévy Flight - thêm bước nhảy Lévy để tăng khả năng exploration
        """
        n_cities = len(coords)
        if n_cities < 2:
            return [0], 0.0
        if n_cities == 2:
            path = np.array([0, 1], dtype=np.int64)
            return path.tolist(), Computing.travel_time(path, coords, v_f, v_AUV)
        
        cities = list(range(1, n_cities))
        
        # -------- init swarm --------
        swarm = [[0] + random.sample(cities, len(cities)) for _ in range(n_particles)]
        velocities = [[] for _ in range(n_particles)]
        
        pbest = [p.copy() for p in swarm]
        pbest_cost = []
        for p in swarm:
            p_arr = np.array(p, dtype=np.int64)
            pbest_cost.append(Computing.travel_time(p_arr, coords, v_f, v_AUV))
        
        # -------- gbest --------
        if init_gbest is not None and len(init_gbest) == n_cities:
            gbest = list(init_gbest)
            gbest_cost = Computing.travel_time(np.array(gbest, np.int64), coords, v_f, v_AUV)
        else:
            idx = int(np.argmin(pbest_cost))
            gbest = pbest[idx].copy()
            gbest_cost = pbest_cost[idx]
        
        # -------- Lévy params --------
        p_levy = 0.15
        beta = 1.5
        max_levy_swaps = 4
        max_velocity_len = n_cities
        
        # --------------------------------
        for t in range(max_iter):
            inertia = 0.7 - 0.5 * (t / max_iter)
            
            for i in range(n_particles):
                xi = swarm[i]
                vi = velocities[i]
                
                # ----- inertia -----
                keep = int(inertia * len(vi))
                v_new = vi[:keep]
                
                # ----- pbest -----
                if random.random() < c1:
                    seq_pb = Pso_routing.get_swap_sequence(xi, pbest[i])
                    if seq_pb:
                        v_new += random.sample(seq_pb, min(2, len(seq_pb)))
                
                # ----- gbest -----
                if random.random() < c2:
                    seq_gb = Pso_routing.get_swap_sequence(xi, gbest)
                    if seq_gb:
                        v_new += random.sample(seq_gb, min(2, len(seq_gb)))
                
                # ----- Lévy flight -----
                if random.random() < p_levy:
                    u = random.gauss(0.0, 1.0)
                    v = random.gauss(0.0, 1.0)
                    step = abs(u) / (abs(v) ** (1.0 / beta))
                    k = min(max_levy_swaps, max(1, int(step)))
                    for _ in range(k):
                        a, b = random.sample(range(1, n_cities), 2)
                        v_new.append((a, b))
                
                # ⛔ CHỐNG velocity rác
                if len(v_new) > max_velocity_len:
                    v_new = random.sample(v_new, max_velocity_len)
                
                # ----- apply -----
                new_x = Pso_routing.apply_velocity(xi, v_new)
                
                # ⛔ SAFETY CHECK (QUAN TRỌNG)
                if new_x is None or len(new_x) != n_cities:
                    continue  # rollback particle
                
                # ✅ KIỂM TRA TÍNH HỢP LỆ CỦA PATH
                if len(set(new_x)) != n_cities or new_x[0] != 0:
                    continue  # Path không hợp lệ
                
                new_x_arr = np.array(new_x, dtype=np.int64)
                new_cost = Computing.travel_time(new_x_arr, coords, v_f, v_AUV)
                
                swarm[i] = new_x
                velocities[i] = v_new
                
                if new_cost < pbest_cost[i]:
                    pbest[i] = new_x.copy()
                    pbest_cost[i] = new_cost
                    
                    if new_cost < gbest_cost:
                        gbest = new_x.copy()
                        gbest_cost = new_cost
            
            if verbose and t % 50 == 0:
                print(f"    [PSO Iter {t:3d}]: Best time = {gbest_cost:.4f}")
        
        if verbose:
            print(f"    [PSO Iter {max_iter:3d}]: Final Best time = {gbest_cost:.4f}")
        
        return gbest, gbest_cost
    
    @staticmethod
    def multi_pso_tsp(coords, v_f=1.2, v_AUV=3.0, n_outer=5, verbose=True, **kwargs):
        """
        Chạy PSO cơ bản nhiều lần để cải thiện hội tụ
        """
        prev_gbest, prev_cost = None, float('inf')
        
        for outer in range(1, n_outer + 1):
            if verbose:
                print(f"   [PSO Outer loop {outer}/{n_outer}]")
            
            gbest, cost = Pso_routing.pso_tsp_3d_time(
                coords, v_f=v_f, v_AUV=v_AUV,
                init_gbest=prev_gbest, verbose=verbose, **kwargs
            )
            
            if cost < prev_cost:
                prev_gbest, prev_cost = gbest, cost
        
        return prev_gbest, prev_cost
    
    @staticmethod
    def multi_pso_tsp_ver2(coords, v_f=1.2, v_AUV=3.0, n_outer=5, verbose=True, **kwargs):
        """
        Chạy PSO với Adaptive Noise nhiều lần
        """
        prev_gbest, prev_cost = None, float('inf')
        
        for outer in range(1, n_outer + 1):
            if verbose:
                print(f"   [PSO Outer loop {outer}/{n_outer}]")
            
            gbest, cost = Pso_routing.pso_tsp_3d_time1(
                coords, v_f=v_f, v_AUV=v_AUV,
                init_gbest=prev_gbest, verbose=verbose, **kwargs
            )
            
            if cost < prev_cost:
                prev_gbest, prev_cost = gbest, cost
        
        return prev_gbest, prev_cost
    
    @staticmethod
    def multi_pso_tsp_ver3(coords, v_f=1.2, v_AUV=3.0, n_outer=5, verbose=True, **kwargs):
        """
        Chạy PSO với Lévy Flight nhiều lần
        """
        prev_gbest, prev_cost = None, float('inf')
        
        for outer in range(1, n_outer + 1):
            if verbose:
                print(f"   [PSO Outer loop {outer}/{n_outer}]")
            
            gbest, cost = Pso_routing.pso_tsp_3d_time2(
                coords, v_f=v_f, v_AUV=v_AUV,
                init_gbest=prev_gbest, verbose=verbose, **kwargs
            )
            
            if cost < prev_cost:
                prev_gbest, prev_cost = gbest, cost
        
        return prev_gbest, prev_cost