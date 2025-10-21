import numpy as np
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

# TÍNH Vs
def compute_vs(p1, p2, v_f, v_AUV):
    """Tính vận tốc tổng hợp v_s giữa 2 vị trí p1, p2"""
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    Lx, Ly, Lz = x2 - x1, y2 - y1, z2 - z1
    L_mag = math.sqrt(Lx**2 + Ly**2 + Lz**2)
    if L_mag == 0:
        return v_AUV

    cos_beta = Lz / L_mag
    cos_beta = np.clip(cos_beta, -1, 1)
    beta = math.acos(cos_beta)

    if abs(cos_beta) < 1e-6:
        cos_beta = 1e-6

    inner = (v_f * cos_beta) / v_AUV
    inner = np.clip(inner, -1, 1)
    angle = beta + math.acos(inner)
    v_s = abs(math.cos(angle) * v_AUV / cos_beta)

    return v_s


# TÍNH THỜI GIAN DI CHUYỂN
def travel_time(path, coords, v_f, v_AUV):
    total_time = 0.0
    for i in range(len(path) - 1):
        p1 = coords[path[i]]
        p2 = coords[path[i + 1]]
        p1_arr = np.array(p1)
        p2_arr = np.array(p2)
        d = np.linalg.norm(p2_arr - p1_arr)
        v_s = compute_vs(tuple(p1_arr.tolist()), tuple(p2_arr.tolist()), v_f, v_AUV)
        if v_s <= 1e-9:
            # avoid division by zero; treat as very large time
            total_time += float('inf')
        else:
            total_time += d / v_s
    p1 = coords[path[-1]]
    p2 = coords[path[0]]
    p1_arr = np.array(p1)
    p2_arr = np.array(p2)
    d = np.linalg.norm(p2_arr - p1_arr)
    v_s = compute_vs(tuple(p1_arr.tolist()), tuple(p2_arr.tolist()), v_f, v_AUV)
    if v_s <= 1e-9:
        total_time += float('inf')
    else:
        total_time += d / v_s
    return total_time


# VẬN TỐC SWARM
def get_swap_sequence(A, B):
    seq = []
    temp = A.copy()
    for i in range(1, len(A)):
        if temp[i] != B[i]:
            j = temp.index(B[i])
            seq.append((i, j))
            temp[i], temp[j] = temp[j], temp[i]
    return seq


def apply_velocity(position, velocity):
    new_pos = position.copy()
    for (i, j) in velocity:
        if i == 0 or j == 0:
            continue
        new_pos[i], new_pos[j] = new_pos[j], new_pos[i]
    return new_pos


class ClusterTSP_GA:
    """GA tối ưu hóa tour qua các cluster centers, sử dụng travel_time/compute_vs để tính mục tiêu (thời gian)."""
    def __init__(self, clusters, ga_params=None):
        # clusters: dict mapping cluster_id -> {"center": (x,y,z), ...}
        self.clusters = clusters
        # tạo list centers với điểm bắt đầu ở index 0
        self.cluster_centers = [(0.0, 0.0, 0.0)] + [clusters[k]["center"] for k in sorted(clusters.keys())]
        self.n = len(self.cluster_centers)

        # default params
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
            'v_f': 0.3,
            'v_AUV': 1.0,
            'verbose': True
        }
        if ga_params:
            defaults.update(ga_params)
        self.params = defaults

        self.best_fitness_history = []
        self.avg_fitness_history = []

    def create_individual(self):
        seq = list(range(1, self.n))
        random.shuffle(seq)
        return [0] + seq

    def create_population(self):
        pop = []
        # add a nearest neighbor as seed
        pop.append(self.nearest_neighbor(0))
        while len(pop) < self.params['pop_size']:
            pop.append(self.create_individual())
        return pop

    def nearest_neighbor(self, start=0):
        unvisited = set(range(1, self.n))
        tour = [start]
        cur = start
        while unvisited:
            nxt = min(unvisited, key=lambda x: np.linalg.norm(np.array(self.cluster_centers[cur]) - np.array(self.cluster_centers[x])))
            tour.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        return tour

    def fitness(self, individual):
        # fitness = 1 / total_time
        total_time = travel_time(individual, self.cluster_centers, self.params['v_f'], self.params['v_AUV'])
        return 1.0 / (total_time + 1e-9)

    def tournament_selection(self, population):
        candidates = random.sample(population, self.params['tournament_size'])
        return max(candidates, key=self.fitness)

    def order_crossover(self, p1, p2):
        # OX on subarray excluding index 0
        sub1 = p1[1:]
        sub2 = p2[1:]
        m = len(sub1)
        if m < 2:
            return p1.copy(), p2.copy()
        a, b = sorted(random.sample(range(m), 2))
        c1 = [-1]*m
        c2 = [-1]*m
        c1[a:b] = sub1[a:b]
        ptr = b
        for x in sub2[b:]+sub2[:b]:
            if x not in c1:
                c1[ptr % m] = x
                ptr += 1
        c2[a:b] = sub2[a:b]
        ptr = b
        for x in sub1[b:]+sub1[:b]:
            if x not in c2:
                c2[ptr % m] = x
                ptr += 1
        return [0]+c1, [0]+c2

    def pmx_crossover(self, p1, p2):
        sub1 = p1[1:]
        sub2 = p2[1:]
        m = len(sub1)
        if m < 2:
            return p1.copy(), p2.copy()
        a, b = sorted(random.sample(range(m), 2))
        c1 = [-1]*m
        c2 = [-1]*m
        c1[a:b] = sub1[a:b]
        c2[a:b] = sub2[a:b]
        mapping1 = {sub1[i]: sub2[i] for i in range(a,b)}
        mapping2 = {sub2[i]: sub1[i] for i in range(a,b)}
        for i in list(range(0,a)) + list(range(b,m)):
            val = sub2[i]
            while val in mapping1:
                val = mapping1[val]
            c1[i] = val
            val = sub1[i]
            while val in mapping2:
                val = mapping2[val]
            c2[i] = val
        return [0]+c1, [0]+c2

    def swap_mutation(self, ind):
        ind = ind.copy()
        if len(ind) > 2:
            i, j = random.sample(range(1, len(ind)), 2)
            ind[i], ind[j] = ind[j], ind[i]
        return ind

    def inversion_mutation(self, ind):
        ind = ind.copy()
        if len(ind) > 2:
            i, j = sorted(random.sample(range(1, len(ind)), 2))
            ind[i:j+1] = list(reversed(ind[i:j+1]))
        return ind

    def two_opt(self, tour):
        improved = True
        best = tour.copy()
        best_time = travel_time(best, self.cluster_centers, self.params['v_f'], self.params['v_AUV'])
        while improved:
            improved = False
            for i in range(1, len(best)-2):
                for j in range(i+1, len(best)-1):
                    cand = best.copy()
                    cand[i:j+1] = list(reversed(cand[i:j+1]))
                    t = travel_time(cand, self.cluster_centers, self.params['v_f'], self.params['v_AUV'])
                    if t < best_time:
                        best = cand
                        best_time = t
                        improved = True
                        break
                if improved:
                    break
        return best

    def evolve(self):
        pop = self.create_population()
        best = max(pop, key=self.fitness)
        best_time = travel_time(best, self.cluster_centers, self.params['v_f'], self.params['v_AUV'])
        for gen in range(self.params['generations']):
            fitnesses = [self.fitness(ind) for ind in pop]
            self.best_fitness_history.append(max(fitnesses))
            self.avg_fitness_history.append(float(np.mean(fitnesses)))
            gen_best = pop[np.argmax(fitnesses)]
            gen_best_time = travel_time(gen_best, self.cluster_centers, self.params['v_f'], self.params['v_AUV'])
            if gen_best_time < best_time:
                best = gen_best.copy()
                best_time = gen_best_time
                if self.params['verbose'] and gen % 50 == 0:
                    print(f"Gen {gen}: new best time = {best_time:.4f} s")

            # elite
            elite_idx = np.argsort(fitnesses)[-self.params['elitism_k']:]
            new_pop = [pop[i].copy() for i in elite_idx]
            while len(new_pop) < self.params['pop_size']:
                p1 = self.tournament_selection(pop)
                p2 = self.tournament_selection(pop)
                if random.random() < self.params['crossover_rate']:
                    if self.params['crossover_type'] == 'OX':
                        c1, c2 = self.order_crossover(p1, p2)
                    else:
                        c1, c2 = self.pmx_crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                if random.random() < self.params['mutation_rate']:
                    if self.params['mutation_type'] == 'swap':
                        c1 = self.swap_mutation(c1)
                    else:
                        c1 = self.inversion_mutation(c1)
                if random.random() < self.params['mutation_rate']:
                    if self.params['mutation_type'] == 'swap':
                        c2 = self.swap_mutation(c2)
                    else:
                        c2 = self.inversion_mutation(c2)

                if self.params['local_search'] and random.random() < 0.1:
                    c1 = self.two_opt(c1)
                    c2 = self.two_opt(c2)

                new_pop.extend([c1, c2])

            pop = new_pop[:self.params['pop_size']]

        return best, best_time

    def print_solution(self, tour, time_total):
        print('\n' + '='*40)
        print(f'Total tour time: {time_total:.4f} s')
        print('Order:')
        for i, idx in enumerate(tour):
            if idx == 0:
                print(f"{i+1}. START (0,0,0)")
            else:
                orig = list(self.clusters.keys())[idx-1]
                print(f"{i+1}. Cluster {orig}: {self.cluster_centers[idx]}")

    def plot_evolution(self):
        if not self.best_fitness_history:
            return
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(self.best_fitness_history, label='best fitness')
        plt.plot(self.avg_fitness_history, label='avg fitness')
        plt.legend()
        plt.xlabel('Gen')
        plt.ylabel('Fitness')
        plt.grid(True)
        plt.subplot(1,2,2)
        best_times = [1.0/f if f>0 else float('inf') for f in self.best_fitness_history]
        avg_times = [1.0/f if f>0 else float('inf') for f in self.avg_fitness_history]
        plt.plot(best_times, label='best time')
        plt.plot(avg_times, label='avg time')
        plt.legend()
        plt.xlabel('Gen')
        plt.ylabel('Time (s)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def main():
    path = "l:\\Tính toán tiến hóa\\IT4906_Project\\IT4906\\output_data_kmeans\\nodes_200.json"
    with open(path, 'r') as f:
        clusters = json.load(f)

    # In ra thông tin giống PSO with EV
    print(f"Đã đọc {len(clusters)} clusters từ file JSON.")

    ga_params = {
        'pop_size': 40,
        'generations': 150,
        'crossover_rate': 0.8,
        'mutation_rate': 0.2,
        'elitism_k': 3,
        'local_search': True,
        'v_f': 0.3,
        'v_AUV': 1.0,
        'verbose': True
    }

    # In ra các tham số GA (giống style PSO with EV)
    print("\n=== GA parameters ===")
    for k, v in ga_params.items():
        print(f"  {k}: {v}")

    ga = ClusterTSP_GA(clusters, ga_params)
    best, best_time = ga.evolve()

    # In kết quả cuối giống PSO
    print("\nFinal best path:", best)
    print(f"Final best total travel time: {best_time:.4f}s")

    ga.print_solution(best, best_time)
    ga.plot_evolution()


if __name__ == '__main__':
    main()

