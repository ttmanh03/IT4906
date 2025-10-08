import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
import time

class ClusterTSP_GA:
    def __init__(self, clusters: Dict, ga_params: Dict = None):
        """
        Khởi tạo GA cho TSP qua centers của các cụm
        
        Args:
            clusters: Dictionary chứa thông tin các cụm
            ga_params: Tham số GA (pop_size, generations, crossover_rate, mutation_rate, elitism_k)
        """
        self.clusters = clusters
        self.cluster_centers = self._extract_centers()
        self.n_clusters = len(clusters)
        self.distance_matrix = self._calculate_distance_matrix()
        
        # Tham số GA mặc định
        default_params = {
            'pop_size': 100,
            'generations': 500,
            'crossover_rate': 0.8,
            'mutation_rate': 0.2,
            'elitism_k': 5,
            'tournament_size': 3,
            'crossover_type': 'OX',  # Order Crossover
            'mutation_type': 'inversion',
            'local_search': True,  # Có áp dụng 2-opt không
            'verbose': True
        }
        
        if ga_params:
            default_params.update(ga_params)
        self.params = default_params
        
        # Lưu trữ lịch sử evolution
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
    def _extract_centers(self) -> List[Tuple[float, float, float]]:
        """Trích xuất tọa độ center từ clusters"""
        centers = []
        for cluster_id in sorted(self.clusters.keys()):
            centers.append(self.clusters[cluster_id]["center"])
        return centers
    
    def _calculate_distance_matrix(self) -> np.ndarray:
        # Tạo ma trận khoảng cách
        n = len(self.cluster_centers)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1, z1 = self.cluster_centers[i]
                    x2, y2, z2 = self.cluster_centers[j]
                    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
                    matrix[i][j] = distance
                    
        return matrix
    
    def calculate_tour_distance(self, tour: List[int]) -> float:
        # Tính tổng khoảng cách 1 tour
        if len(tour) != self.n_clusters:
            raise ValueError(f"Tour phải có đúng {self.n_clusters} clusters")
            
        total_distance = 0
        for i in range(len(tour)):
            current = tour[i]
            next_city = tour[(i + 1) % len(tour)]  # Quay về điểm đầu
            total_distance += self.distance_matrix[current][next_city]
            
        return total_distance
    
    def fitness(self, individual: List[int]) -> float:
        # Hàm fitness: max
        distance = self.calculate_tour_distance(individual)
        return 1.0 / (distance + 1e-6)  # Tránh chia cho 0
    
    def create_individual(self) -> List[int]:
        # Tạo 1 tour random
        return list(np.random.permutation(self.n_clusters))
    
    def create_population(self) -> List[List[int]]:
        # Tạo quần thể ban đầu
        population = []
        
        # Một số cá thể được tạo bằng nearest neighbor heuristic
        for start in range(min(5, self.n_clusters)):
            nn_tour = self.nearest_neighbor_heuristic(start)
            population.append(nn_tour)
        
        # Phần còn lại random
        while len(population) < self.params['pop_size']:
            population.append(self.create_individual())
            
        return population
    
    def nearest_neighbor_heuristic(self, start: int = 0) -> List[int]:
        # Heuristic nearest neighbor
        unvisited = set(range(self.n_clusters))
        tour = [start]
        unvisited.remove(start)
        
        current = start
        while unvisited:
            nearest = min(unvisited, key=lambda x: self.distance_matrix[current][x])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
            
        return tour
    
    def tournament_selection(self, population: List[List[int]]) -> List[int]:
        """Tournament selection"""
        tournament = random.sample(population, self.params['tournament_size'])
        return max(tournament, key=self.fitness)
    
    def order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        # Order Crossover (OX)
        size = len(parent1)
        
        # Chọn 2 điểm cắt ngẫu nhiên
        start, end = sorted(random.sample(range(size), 2))
        
        # Tạo offspring 1
        child1 = [-1] * size
        child1[start:end] = parent1[start:end]
        
        # Điền các phần tử còn lại từ parent2
        pointer = end
        for city in parent2[end:] + parent2[:end]:
            if city not in child1:
                child1[pointer % size] = city
                pointer += 1
        
        # Tương tự cho offspring 2
        child2 = [-1] * size
        child2[start:end] = parent2[start:end]
        
        pointer = end
        for city in parent1[end:] + parent1[:end]:
            if city not in child2:
                child2[pointer % size] = city
                pointer += 1
        
        return child1, child2
    
    def pmx_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        # Lai tương hợp bộ phận
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child1 = [-1] * size
        child2 = [-1] * size
        
        # Copy mapping sections
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        
        # Create mapping dictionaries
        mapping1 = {parent1[i]: parent2[i] for i in range(start, end)}
        mapping2 = {parent2[i]: parent1[i] for i in range(start, end)}
        
        # Fill remaining positions
        for i in list(range(start)) + list(range(end, size)):
            # For child1
            val = parent2[i]
            while val in mapping1:
                val = mapping1[val]
            child1[i] = val
            
            # For child2
            val = parent1[i]
            while val in mapping2:
                val = mapping2[val]
            child2[i] = val
            
        return child1, child2
    
    def swap_mutation(self, individual: List[int]) -> List[int]:
        # Đổi chỗ
        mutated = individual.copy()
        i, j = random.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated
    
    def inversion_mutation(self, individual: List[int]) -> List[int]:
        # Đảo đoạn
        mutated = individual.copy()
        start, end = sorted(random.sample(range(len(mutated)), 2))
        mutated[start:end+1] = reversed(mutated[start:end+1])
        return mutated
    
    def two_opt_local_search(self, tour: List[int]) -> List[int]:
        """2-opt local search để cải thiện tour"""
        improved = True
        current_tour = tour.copy()
        
        while improved:
            improved = False
            current_distance = self.calculate_tour_distance(current_tour)
            
            for i in range(len(current_tour)):
                for j in range(i+2, len(current_tour)):
                    # Tạo tour mới bằng cách đảo đoạn từ i+1 đến j
                    new_tour = current_tour.copy()
                    new_tour[i+1:j+1] = reversed(new_tour[i+1:j+1])
                    
                    new_distance = self.calculate_tour_distance(new_tour)
                    if new_distance < current_distance:
                        current_tour = new_tour
                        improved = True
                        break
                        
                if improved:
                    break
                    
        return current_tour
    
    def evolve(self) -> Tuple[List[int], float]:
        # Chạy GA
        if self.params['verbose']:
            print(f"{self.n_clusters} cluster centers")
            print(f"Tham số: {self.params}")
        
        # Khởi tạo quần thể
        population = self.create_population()
        
        # Đánh giá fitness ban đầu
        best_individual = max(population, key=self.fitness)
        best_distance = self.calculate_tour_distance(best_individual)
        
        start_time = time.time()
        
        for generation in range(self.params['generations']):
            # Đánh giá fitness cho toàn bộ quần thể
            fitness_scores = [self.fitness(ind) for ind in population]
            
            # Lưu thống kê
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            # Tìm cá thể tốt nhất thế hệ này
            gen_best = population[np.argmax(fitness_scores)]
            gen_best_distance = self.calculate_tour_distance(gen_best)
            
            if gen_best_distance < best_distance:
                best_individual = gen_best.copy()
                best_distance = gen_best_distance
                
                if self.params['verbose'] and generation % 50 == 0:
                    print(f"Gen {generation}: New best distance = {best_distance:.3f}")
            
            # Elitism - giữ lại k cá thể tốt nhất
            elite_indices = np.argsort(fitness_scores)[-self.params['elitism_k']:]
            new_population = [population[i].copy() for i in elite_indices]
            
            # Tạo thế hệ mới
            while len(new_population) < self.params['pop_size']:
                # Selection
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                # Crossover
                if random.random() < self.params['crossover_rate']:
                    if self.params['crossover_type'] == 'OX':
                        child1, child2 = self.order_crossover(parent1, parent2)
                    else:  # PMX
                        child1, child2 = self.pmx_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if random.random() < self.params['mutation_rate']:
                    if self.params['mutation_type'] == 'swap':
                        child1 = self.swap_mutation(child1)
                    else:  # inversion
                        child1 = self.inversion_mutation(child1)
                        
                if random.random() < self.params['mutation_rate']:
                    if self.params['mutation_type'] == 'swap':
                        child2 = self.swap_mutation(child2)
                    else:  # inversion
                        child2 = self.inversion_mutation(child2)
                
                # Local search (optional)
                if self.params['local_search'] and random.random() < 0.1:  # 10% chance
                    child1 = self.two_opt_local_search(child1)
                    child2 = self.two_opt_local_search(child2)
                
                new_population.extend([child1, child2])
            
            # Cắt bớt nếu vượt quá pop_size
            population = new_population[:self.params['pop_size']]
        
        elapsed_time = time.time() - start_time
        
        if self.params['verbose']:
            print(f"\nKết thúc GA sau {elapsed_time:.2f}s")
            print(f"Best tour distance: {best_distance:.3f}")
            print(f"Best tour: {best_individual}")
        
        return best_individual, best_distance
    
    # Kết quả
    def print_solution(self, tour: List[int], distance: float):
        print("\n" + "="*50)
        print("KẾT QUẢ TSP CHO CÁC CLUSTER CENTERS")
        print("="*50)
        print(f"Tổng khoảng cách tour: {distance:.3f}")
        print(f"Số cluster centers: {len(tour)}")
        print("\nThứ tự đi qua các clusters:")
        
        for i, cluster_idx in enumerate(tour):
            center = self.cluster_centers[cluster_idx]
            cluster_head = self.clusters[cluster_idx].get("cluster_head", "N/A")
            print(f"{i+1}. Cluster {cluster_idx}: Center{center} (Head: {cluster_head})")
            
        # Quay về điểm đầu
        start_center = self.cluster_centers[tour[0]]
        print(f"{len(tour)+1}. Quay về Cluster {tour[0]}: Center{start_center}")
        
        print(f"\nChi tiết khoảng cách giữa các bước:")
        total_check = 0
        for i in range(len(tour)):
            current = tour[i]
            next_cluster = tour[(i+1) % len(tour)]
            step_distance = self.distance_matrix[current][next_cluster]
            total_check += step_distance
            print(f"  Cluster {current} -> Cluster {next_cluster}: {step_distance:.3f}")
        
        print(f"\nKiểm tra tổng: {total_check:.3f}")
    
    def plot_evolution(self):
        """Vẽ biểu đồ quá trình evolution"""
        if not self.best_fitness_history:
            print("Chưa có dữ liệu evolution để vẽ")
            return
            
        plt.figure(figsize=(12, 5))
        
        # Fitness evolution
        plt.subplot(1, 2, 1)
        plt.plot(self.best_fitness_history, label='Best Fitness', color='red')
        plt.plot(self.avg_fitness_history, label='Average Fitness', color='blue')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution')
        plt.legend()
        plt.grid(True)
        
        # Distance evolution (convert fitness back to distance)
        plt.subplot(1, 2, 2)
        best_distances = [1.0/f for f in self.best_fitness_history]
        avg_distances = [1.0/f for f in self.avg_fitness_history]
        plt.plot(best_distances, label='Best Distance', color='red')
        plt.plot(avg_distances, label='Average Distance', color='blue')
        plt.xlabel('Generation')
        plt.ylabel('Tour Distance')
        plt.title('Distance Evolution')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Chạy thử
def main():
    
    with open("l:\\Tính toán tiến hóa\\IT4906_Project\\IT4906\\output_data\\nodes_500.json", "r") as file:
        clusters = json.load(file)
    
    # Param GA
    ga_params = {
        'pop_size': 50,
        'generations': 200,
        'crossover_rate': 0.8,
        'mutation_rate': 0.2,
        'elitism_k': 3,
        'crossover_type': 'OX',
        'mutation_type': 'inversion',
        'local_search': True,
        'verbose': True
    }
    
    # Chạy GA
    tsp_ga = ClusterTSP_GA(clusters, ga_params)
    best_tour, best_distance = tsp_ga.evolve()
    
    # In kết quả
    tsp_ga.print_solution(best_tour, best_distance)
    
    # Vẽ biểu đồ evolution
    tsp_ga.plot_evolution()
    
    return tsp_ga, best_tour, best_distance

if __name__ == "__main__":
    # Chạy ví dụ
    tsp_ga, tour, distance = main()