import sys
import os
import random
import numpy as np
from numba import njit
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from compute import Computing

class ClusterTSP_GA:
    def __init__(self, clusters, ga_params=None):
        self.clusters = clusters
        sorted_keys = sorted(clusters.keys(), key=lambda x: int(x))
        self.index_to_ch = [None]
        self.cluster_centers = [(0.0, 0.0, 0.0)]
        for k in sorted_keys:
            c = clusters[k]['center']
            self.cluster_centers.append(tuple(c))
            self.index_to_ch.append(clusters[k].get('cluster_head', None))
        self.n = len(self.cluster_centers)
        defaults = {
            'pop_size': 50,
            'generations': 200,
            'crossover_rate': 0.8,
            'mutation_rate': 0.2,
            'elitism_k': 3,
            'tournament_size': 3,
            'crossover_type': 'OX',
            'mutation_type': 'inversion',
            'local_search': True,
            'v_f': 1.2,
            'v_AUV': 3.0,
            'verbose': False
        }
        if ga_params:
            defaults.update(ga_params)
        self.params = defaults
        self.best_fitness_history = []
    
    def create_individual(self):
        if self.n < 2:  # Chỉ có depot
            return [0]
        seq = list(range(1, self.n))
        random.shuffle(seq)
        return [0] + seq
    
    def create_population(self):
        return [self.create_individual() for _ in range(self.params['pop_size'])]
    
    def fitness(self, ind):
        total_time = Computing.travel_time(ind, self.cluster_centers, self.params['v_f'], self.params['v_AUV'])
        return 1.0 / (total_time + 1e-9)
    
    def tournament_selection(self, population):
        return max(random.sample(population, self.params['tournament_size']), key=self.fitness)
    
    def order_crossover(self, p1, p2):
        sub1, sub2 = p1[1:], p2[1:]
        # Kiểm tra trường hợp đặc biệt
        if len(sub1) < 2:
            return p1.copy(), p2.copy()
        a, b = sorted(random.sample(range(len(sub1)), 2))
        c1, c2 = [-1]*len(sub1), [-1]*len(sub1)
        c1[a:b], c2[a:b] = sub1[a:b], sub2[a:b]
        ptr = b
        for x in sub2[b:]+sub2[:b]:
            if x not in c1:
                c1[ptr % len(sub1)] = x
                ptr += 1
        ptr = b
        for x in sub1[b:]+sub1[:b]:
            if x not in c2:
                c2[ptr % len(sub2)] = x
                ptr += 1
        return [0]+c1, [0]+c2
    
    def inversion_mutation(self, ind):
        i, j = sorted(random.sample(range(1, len(ind)), 2))
        ind[i:j+1] = list(reversed(ind[i:j+1]))
        return ind
    
    def evolve(self):
        """
        Evolve với logging chi tiết và lưu history
        Returns: best_path, mapped_path, best_time, history
        """
        # Khởi tạo quần thể
        pop = self.create_population()
        best = max(pop, key=self.fitness)
        best_time = Computing.travel_time(best, self.cluster_centers, self.params['v_f'], self.params['v_AUV'])
        
        # Lưu history
        history = [best_time]
        
        # In thông tin ban đầu
        if self.params['verbose']:
            print(f"   [GA] Khởi tạo:")
            print(f"        Population size: {self.params['pop_size']}")
            print(f"        Generations: {self.params['generations']}")
            print(f"        Crossover rate: {self.params['crossover_rate']}")
            print(f"        Mutation rate: {self.params['mutation_rate']}")
            print(f"        Initial best time: {best_time:.4f}s\n")
        
        # Biến đếm cho statistics
        improvements = 0
        last_improvement_gen = 0
        
        # Vòng lặp chính
        for gen in range(self.params['generations']):
            # Tính fitness cho tất cả cá thể
            fitnesses = [self.fitness(ind) for ind in pop]
            
            # Tìm cá thể tốt nhất thế hệ này
            best_gen_idx = np.argmax(fitnesses)
            best_gen = pop[best_gen_idx]
            gen_best_time = Computing.travel_time(best_gen, self.cluster_centers, self.params['v_f'], self.params['v_AUV'])
            gen_avg_time = np.mean([Computing.travel_time(ind, self.cluster_centers, self.params['v_f'], self.params['v_AUV']) 
                                   for ind in pop])
            gen_worst_time = min([Computing.travel_time(ind, self.cluster_centers, self.params['v_f'], self.params['v_AUV']) 
                                  for ind in pop])
            
            # Cập nhật best nếu tốt hơn
            improved = False
            if gen_best_time < best_time:
                improvement_percent = (best_time - gen_best_time) / best_time * 100
                best, best_time = best_gen.copy(), gen_best_time
                improvements += 1
                last_improvement_gen = gen
                improved = True
                
                if self.params['verbose']:
                    print(f"   [Gen {gen:4d}] 🎯 NEW BEST! Time: {best_time:.4f}s (improved {improvement_percent:.2f}%)")
            
            # Lưu history
            history.append(best_time)
            
            # In log mỗi N generations
            log_interval = max(1, self.params['generations'] // 10)  # In 10 lần
            if self.params['verbose'] and (gen % log_interval == 0 or gen == self.params['generations'] - 1):
                if not improved:  # Không in lại nếu vừa in "NEW BEST"
                    print(f"   [Gen {gen:4d}] Best: {best_time:.4f}s | Gen Best: {gen_best_time:.4f}s | "
                          f"Avg: {gen_avg_time:.4f}s | Worst: {gen_worst_time:.4f}s")
            
            # Chọn elite
            elite_idx = np.argsort(fitnesses)[-self.params['elitism_k']:]
            new_pop = [pop[i].copy() for i in elite_idx]
            
            # Tạo thế hệ mới
            while len(new_pop) < self.params['pop_size']:
                # Selection
                p1 = self.tournament_selection(pop)
                p2 = self.tournament_selection(pop)
                
                # Crossover
                if random.random() < self.params['crossover_rate']:
                    c1, c2 = self.order_crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Mutation
                if random.random() < self.params['mutation_rate']:
                    c1 = self.inversion_mutation(c1)
                if random.random() < self.params['mutation_rate']:
                    c2 = self.inversion_mutation(c2)
                
                new_pop += [c1, c2]
            
            pop = new_pop[:self.params['pop_size']]
        
        # In thống kê cuối cùng
        if self.params['verbose']:
            print(f"\n   [GA] Kết thúc:")
            print(f"        Final best time: {best_time:.4f}s")
            print(f"        Initial time: {history[0]:.4f}s")
            print(f"        Total improvement: {(history[0] - best_time):.4f}s ({(history[0] - best_time) / history[0] * 100:.2f}%)")
            print(f"        Number of improvements: {improvements}")
            print(f"        Last improvement at generation: {last_improvement_gen}")
            print(f"        Convergence rate: {improvements / self.params['generations'] * 100:.2f}%")
        
        # Map path về cluster head IDs
        mapped_path = ['O' if idx == 0 else self.index_to_ch[idx] for idx in best]
        
        return best, mapped_path, best_time, history

print("GA class with detailed logging and history tracking loaded")