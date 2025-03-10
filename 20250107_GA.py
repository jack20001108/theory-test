from pyDOE2 import lhs
import numpy as np
import time
from scipy.stats import poisson
from randomgen import Xoroshiro128
from deap import base, creator, tools, algorithms


# "零件"庫存與缺貨計算
def evaluator(demand, order, leadtime, begin_inventory, bom):
    # 矩陣形狀
    num_simulations, num_products, num_periods = demand.shape  # 模擬次數K, 產品數N, 時間期數T
    num_material = begin_inventory.shape[0]  # 零件數量M

    # 初始庫存矩陣
    inventory = np.zeros((num_simulations, num_material, num_periods + 1))  # K*M*(T+1)
    inventory[:, :, 0] = begin_inventory

    # 計算零件需求
    component_demand = bom[:, :, None] * demand[:, None, :, :]
    cumulate_demand = np.cumsum(component_demand.sum(axis=2), axis=2)  # K*M*T

    # 計算累積到貨
    arrival_order = np.roll(order, shift=leadtime, axis=1)
    arrival_order[:, :leadtime] = 0
    arrival_order = np.expand_dims(arrival_order, axis=0).repeat(num_simulations, axis=0)
    cumulate_arrival = np.cumsum(arrival_order, axis=2)  # K*M*T

    # 計算每期庫存
    inventory[:, :, 1:] = begin_inventory[None, :, None] + cumulate_arrival - cumulate_demand

    # 計算缺貨
    shortage = np.where(inventory[:, :, 1:] < 0, -inventory[:, :, 1:], 0)

    # 計算未分配需求
    reverse_cumulative_demand = np.cumsum(component_demand[:, ::-1, :, :], axis=1)[:, ::-1, :, :]
    adjusted_reverse_cumulative_demand = np.concatenate(
        [reverse_cumulative_demand[:, 1:, :, :], np.zeros((num_simulations, 1, num_products, num_periods))], axis=1
    )
    reverse_inventory = inventory[:, :, 1:, None]
    reverse2_inventory = reverse_inventory.transpose(0, 1, 3, 2)

    unclaim_inventory = reverse2_inventory + adjusted_reverse_cumulative_demand

    # 計算零件滿足率
    aligned_bom_demand = component_demand.transpose(0, 3, 1, 2)
    shortage_com = np.where(
        unclaim_inventory.transpose(0, 3, 1, 2) < 0,
        np.minimum(-unclaim_inventory.transpose(0, 3, 1, 2), aligned_bom_demand),
        0,
    )
    safe_aligned_bom_demand = np.where(aligned_bom_demand == 0, 1e-10, aligned_bom_demand)
    com_fill_rate = 1 - (shortage_com / safe_aligned_bom_demand)

    return inventory, shortage, unclaim_inventory, com_fill_rate


# 拉丁超立方抽樣生成泊松分布需求矩陣
def generate_lhs_poisson_demand(num_simulations, num_products, num_periods, mean_demands, seed=None):
    total_samples = num_simulations * num_periods
    rng = np.random.Generator(Xoroshiro128(seed)) if seed is not None else np.random.Generator(Xoroshiro128())
    lhs_samples = lhs(num_products, samples=total_samples)
    uniform_random_samples = rng.uniform(size=lhs_samples.shape)
    lhs_samples = (lhs_samples + uniform_random_samples) / total_samples

    poisson_samples = np.zeros_like(lhs_samples)
    for i, mean_demand in enumerate(mean_demands):
        poisson_samples[:, i] = poisson.ppf(lhs_samples[:, i], mu=mean_demand)

    demand_matrix = poisson_samples.reshape(num_simulations, num_periods, num_products).transpose(0, 2, 1)
    return demand_matrix.astype(int)


# Fitness 函數
def fitness(order, leadtime, begin_inventory, bom, target_fill_rate):
    demand_matrix = generate_lhs_poisson_demand(num_simulations, num_products, num_periods, mean_demands, seed=42)
    inventory, shortage, unclaim_inventory, com_fill_rate = evaluator(demand_matrix, order, leadtime, begin_inventory, bom)

    product_fill_rate_by_period = np.prod(com_fill_rate, axis=2).mean(axis=0)
    average_product_fill_rate = np.mean(product_fill_rate_by_period, axis=0)

    diff = np.abs(average_product_fill_rate - target_fill_rate)
    penalty = 0  # 可以加入限制式違反的懲罰
    weights = np.array([0.33, 0.33, 0.33])
    fitness_value = -np.sum(diff * weights) + penalty
    return fitness_value


# 測試參數
num_simulations = 1000  # 模擬次數
num_products = 3        # 產品數量
num_periods = 5         # 時間期數
mean_demands = [60, 30, 50]  # 每個產品泊松分布的平均需求
seed = 42              # 隨機數生成器的種子
leadtime = 2           # 產品到貨的 Lead Time
begin_inventory = np.array([100, 50, 70])  # 初始庫存
bom = np.array([
    [1, 0, 1],
    [2, 1, 0],
    [0, 1, 1],
    [1, 1, 0]
])
target_fill_rate = np.array([0.95, 0.90, 0.85])  # 目標滿足率


# 基因演算法設置
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_order", np.random.randint, 10, 50)  # 隨機生成每期訂單數量
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_order, n=num_periods)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", lambda x: fitness(np.array(x).reshape((1, -1)), leadtime, begin_inventory, bom, target_fill_rate))
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# 運行 GA
population = toolbox.population(n=50)
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=100, verbose=True)
