
import sys
import os
import numpy as np
from numba import njit
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from compute import Computing


class Greedy:
    """
    Class chứa thuật toán Greedy để giải quyết bài toán TSP (Traveling Salesman Problem)
    """
    
    @staticmethod
    def greedy_tsp(coords, v_f=1.2, v_AUV=3.0):
        """
        Thuật toán Greedy cho TSP - tại mỗi bước chọn node gần nhất chưa thăm
        
        Args:
            coords: array - tọa độ các node (không bao gồm base station)
            v_f: float - vận tốc dòng chảy
            v_AUV: float - vận tốc AUV
            
        Returns:
            tuple: (best_path, best_time)
                - best_path: list - đường đi tốt nhất (bắt đầu và kết thúc tại BS - index 0)
                - best_time: float - tổng thời gian di chuyển
        """
        # Thêm Base Station vào đầu danh sách coords
        base_station = np.array([[200, 200, 400]])
        coords = np.vstack([base_station, coords])
        
        n = len(coords)
        if n < 2:
            return list(range(n)), 0.0
        
        # Biến lưu kết quả tốt nhất
        argmintime = None  # Đường đi có thời gian ngắn nhất
        best_time = float('inf')
        
        # Tập các node cần thăm (không bao gồm base station - node 0)
        TN_s = set(range(1, n))
        
        def function_dp(current_node, remaining_nodes, current_path, current_time):
            """
            Hàm đệ quy tìm đường đi theo chiến lược Greedy
            
            Args:
                current_node: int - node hiện tại
                remaining_nodes: set - tập các node chưa thăm
                current_path: list - đường đi hiện tại
                current_time: float - tổng thời gian hiện tại
            """
            nonlocal argmintime, best_time
            
            # Đã thăm hết tất cả nodes
            if len(remaining_nodes) == 0:
                # Tính thời gian quay về BS (điểm 0)
                path_return = [current_node, 0]
                time_return = Computing.travel_time(
                    np.array(path_return), coords, v_f, v_AUV
                )
                total_time = current_time + time_return
                
                # Cập nhật kết quả tốt nhất
                if total_time < best_time:
                    best_time = total_time
                    argmintime = current_path + [0]  # Thêm điểm quay về BS
                
                return
            
            # Thử tất cả các node còn lại - chọn node gần nhất (Greedy)
            mintime = float('inf')
            best_next_node = None
            
            for next_node in remaining_nodes:
                # Tính thời gian từ current_node đến next_node
                path_segment = [current_node, next_node]
                t_i = Computing.travel_time(
                    np.array(path_segment), coords, v_f, v_AUV
                )
                
                # Chọn node có thời gian di chuyển ngắn nhất (Greedy)
                if t_i < mintime:
                    mintime = t_i
                    best_next_node = next_node
            
            # Di chuyển đến node tốt nhất
            if best_next_node is not None:
                new_remaining = remaining_nodes - {best_next_node}
                new_path = current_path + [best_next_node]
                new_time = current_time + mintime
                
                # Gọi đệ quy
                function_dp(best_next_node, new_remaining, new_path, new_time)
        
        # Bắt đầu từ base station (node 0)
        function_dp(0, TN_s, [0], 0)
        
        return argmintime, best_time