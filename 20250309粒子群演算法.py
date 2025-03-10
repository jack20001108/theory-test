import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import norm

# 設定參數
num_products = 3
num_materials = 4
num_periods = 5  
leadtime = 1
inventory_cost_per_unit = 2
num_generations = 100
target_fill_rate = 0.95
num_particles = 50  # 粒子數量

bom = np.array([
    [1, 0, 1],
    [2, 1, 0],
    [0, 1, 1],
    [1, 1, 0]
])

# 評估函數
def evaluator(demand, order, leadtime):  
    num_product, num_period = demand.shape
    inventory = np.zeros((num_materials, num_period))
    begin_inventory = np.tile(order[:, 0][:, None], (1, num_periods))
    bom_demand = bom[:, :, None] * demand
    material_demand = bom_demand.transpose(2, 0, 1)
    cumulative_demand = np.cumsum(bom @ demand, axis=1)
    real_order = order[:, 1:]
    arrival_order = np.roll(real_order, shift=leadtime, axis=1)
    arrival_order[:, :leadtime] = 0
    cumulative_arrival = np.cumsum(arrival_order, axis=1)
    inventory = begin_inventory + cumulative_arrival - cumulative_demand
    real_stock = np.where(inventory < 0, 0, inventory)
    return material_demand, real_stock

def fitness(order):
    sample_size = 100
    min_demand, max_demand = 10, 50
    demand_samples = np.random.randint(min_demand, max_demand + 1, size=(sample_size, num_products, num_periods))
    cumulative_fill_rate = []
    real_stock_list = []

    for demand in demand_samples:
        material_demand, real_stock = evaluator(demand, order, leadtime)
        real_stock_list.append(real_stock)

    real_stock_array = np.array(real_stock_list)
    real_stock_mean = np.mean(real_stock_array, axis=0)

    # ⚡ 修正：增加未滿足需求的懲罰
    inventory_cost = np.sum(real_stock_mean) * inventory_cost_per_unit
    penalty = np.sum(np.maximum(0, target_fill_rate - np.mean(real_stock_mean)))

    return inventory_cost + penalty  # 避免 fitness = 0


# PSO 參數
topology_size = num_particles  # 粒子數量
inertia_weight = 0.7
c1 = 1.5  # 個體最佳解權重
c2 = 1.5  # 全局最佳解權重

# 初始化粒子群
particles = np.random.randint(0, 500, (topology_size, num_materials, num_periods + 1))
velocities = np.random.uniform(-1, 1, (topology_size, num_materials, num_periods + 1))
personal_best = np.copy(particles)
personal_best_fitness = np.array([fitness(p) for p in personal_best])
global_best = personal_best[np.argmin(personal_best_fitness)]
global_best_fitness = np.min(personal_best_fitness)

# 儲存最佳適應值
generations = []
fitness_history = []

# 進行 PSO 迭代
for gen in range(num_generations):
    for i in range(topology_size):
        velocities[i] = (
            inertia_weight * velocities[i] +
            c1 * np.random.rand() * (personal_best[i] - particles[i]) +
            c2 * np.random.rand() * (global_best - particles[i])
        )
        particles[i] = np.clip(particles[i] + velocities[i], 0, 500)
        particles[i] = particles[i].astype(int)
        fitness_value = fitness(particles[i])
        if fitness_value < personal_best_fitness[i]:
            personal_best[i] = particles[i]
            personal_best_fitness[i] = fitness_value
        if fitness_value < global_best_fitness:
            global_best = particles[i]
            global_best_fitness = fitness_value
    generations.append(gen)
    fitness_history.append(global_best_fitness)
    print(f"Generation {gen}: Best Fitness = {global_best_fitness}")

# 結果分析
plt.plot(generations, fitness_history, marker='o', linestyle='-', label='Best Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('PSO Convergence')
plt.legend()
plt.grid(True)
plt.show()

print("Best Solution:")
print(global_best)
