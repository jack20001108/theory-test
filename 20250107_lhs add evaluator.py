from pyDOE2 import lhs
import numpy as np
import time
from scipy.stats import poisson
from randomgen import Xoroshiro128

# 拉丁超立方抽樣生成泊松分布需求矩陣
def generate_lhs_poisson_demand(num_simulations, num_products, num_periods, mean_demands, seed=None):
    total_samples = num_simulations * num_periods
    rng = np.random.default_rng(seed)  # 默認隨機生成器

    # 拉丁超立方抽樣
    lhs_samples = lhs(num_products, samples=total_samples)
    uniform_random_samples = rng.uniform(size=lhs_samples.shape)  # 均勻隨機抖動
    lhs_samples = (lhs_samples + uniform_random_samples) / total_samples

    # 將均勻樣本映射為泊松分布
    poisson_samples = np.zeros_like(lhs_samples)
    for i, mean_demand in enumerate(mean_demands):
        poisson_samples[:, i] = poisson.ppf(lhs_samples[:, i], mu=mean_demand)

    # 重塑需求矩陣
    demand_matrix = poisson_samples.reshape(num_simulations, num_periods, num_products).transpose(0, 2, 1)

    # Debug 打印第一個產品的需求
    print(f"LHS samples for Product 1: {lhs_samples[:, 0]}")
    print(f"Poisson samples for Product 1: {poisson.ppf(lhs_samples[:, 0], mu=mean_demands[0])}")
    
    return demand_matrix.astype(int)





# "零件"庫存與缺貨計算
def evaluator(demand, order, leadtime, begin_inventory, bom):
    #矩陣形狀 (define metrix shape)
    num_simulations, num_products, num_periods = demand.shape #三維(模擬次數K,產品數量N,期數T)
    num_material = begin_inventory.shape[0] #零件數量 

    #初始庫存矩陣
    inventory = np.zeros((num_simulations, num_material, num_periods + 1))  # K*M*(T+1) 
    inventory[:, :, 0] = begin_inventory #初始庫存
    print("demand：", demand)    
    #計算零件需求
    component_demand = bom[:, :, None] * demand[:, None, :, :] # K*M*N*T
    print("component_demand：", component_demand)
    #計算累積需求
    cumulate_demand = np.cumsum(component_demand.sum(axis=2), axis=2)  # K*M*T
    print("cumulate_demand：", cumulate_demand)

    #計算累積到貨
    arrival_order = np.roll(order, shift=leadtime, axis=1)  #滾動補貨前置期
    arrival_order[:, :leadtime] = 0  #前 leadtime 期無到貨
    arrival_order = np.expand_dims(arrival_order, axis=0).repeat(num_simulations, axis=0)  # 擴展至K模擬情境
    cumulate_arrival = np.cumsum(arrival_order, axis=2)  # K*M*T
    print("cumulate_arrival：", cumulate_arrival)
    #計算每期庫存
    inventory[:, :, 1:] = begin_inventory[None, :, None] + cumulate_arrival - cumulate_demand

    #計算缺貨
    shortage = np.where(inventory[:, :, 1:] < 0, -inventory[:, :, 1:], 0)  #該期缺貨數量

    #計算未分配需求
    shift_bom_demand=np.roll(component_demand,shift=-1,axis=1) 
    shift_bom_demand[:,-1,:,:]=0

    reverse_cumulative_demand = np.cumsum(component_demand[:, ::-1, :, :], axis=1)[:, ::-1, :, :]  #反累積需求
    print("reverse_cumulative_demand：",reverse_cumulative_demand)
    #刪除第0個矩陣
    modified_cumulative_demand = reverse_cumulative_demand[:, 1:, :, :]

    #補上零矩陣
    zero_matrix = np.zeros((num_simulations, 1, num_products, num_periods))
    adjusted_reverse_cumulative_demand = np.concatenate([modified_cumulative_demand, zero_matrix],axis=1 )
    print("adjusted_reverse_cumulative_demand：",adjusted_reverse_cumulative_demand)
    reverse_inventory=inventory[:, :, 1:,None]
    reverse2_inventory=reverse_inventory.transpose(0,1,3,2)
    print("reverse2_inventory：",reverse2_inventory)

    unclaim_inventory = (
        reverse2_inventory
        + adjusted_reverse_cumulative_demand  #未分配庫存
    )
    print("unclaim_inventory：",unclaim_inventory)

    #每時間點，零件分配角度看是否缺貨(矩陣代表時間T=1,2,...,t)
    reshape_unclaim=unclaim_inventory.transpose(0,3,1,2)
    print("reshape_unclaim：",reshape_unclaim)
    #每個產品，所需零件在各期間是否缺貨 (矩陣代表產品N)
    reshape_to_product=unclaim_inventory.transpose(0,2,1,3)
    print("reshape_to_product：",reshape_to_product)    


    #計算零件滿足率
    aligned_bom_demand = component_demand.transpose(0,3,1,2)
    shortage_com = np.where(
        reshape_unclaim < 0,
        np.minimum(-reshape_unclaim, aligned_bom_demand),
        0
    )

    # 避免分母為零
    safe_aligned_bom_demand = np.where(aligned_bom_demand == 0, 1e-10, aligned_bom_demand)

    # 計算 fill rate
    com_fill_rate = 1 - (shortage_com / safe_aligned_bom_demand)

    return inventory, shortage, unclaim_inventory, com_fill_rate

# 需求模擬整合至 fitness 流程
def fitness(order, leadtime, begin_inventory, bom, target_fill_rate):
    start_time = time.time()

    # 使用 LHS 生成需求
    lower_bounds = [5, 20, 40]
    upper_bounds = [15, 40, 70]
    demand_matrix = generate_lhs_poisson_demand(num_simulations, num_products, num_periods, lower_bounds, upper_bounds)

    # 傳入 evaluator
    inventory, shortage, unclaim_inventory, com_fill_rate = evaluator(demand_matrix, order, leadtime, begin_inventory, bom)

    # 計算產品層級的平均滿足率
    product_fill_rate_by_period = np.prod(com_fill_rate, axis=2).mean(axis=0)  # 每期平均滿足率
    average_product_fill_rate = np.mean(product_fill_rate_by_period, axis=0)

    # 計算目標滿足率差異
    diff = np.abs(average_product_fill_rate - target_fill_rate)
    weights = np.array([0.33, 0.33, 0.33])  # 三個產品的權重
    fitness_value = -np.sum(diff * weights)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Fitness value: {fitness_value}")
    print(f"Execution time: {execution_time:.4f} seconds")
    return fitness_value

# 測試參數

num_simulations = 10
num_products = 3
num_periods = 5
mean_demands = [50, 60, 70]  # 平均需求
seed = 42

demand_matrix = generate_lhs_poisson_demand(num_simulations, num_products, num_periods, mean_demands, seed)

bom = np.array([
    [1, 0, 1],
    [2, 1, 0],
    [0, 1, 1],
    [1, 1, 0]
])
order = np.array([
    [10, 20, 30, 10, 5],
    [15, 25, 20, 15, 10],
    [12, 18, 15, 12, 10],
    [8, 10, 15, 20, 10]
])
begin_inventory = np.array([50, 40, 30, 20])
leadtime = 1
target_fill_rate = np.array([0.95, 0.90, 0.85])  # 目標滿足率


# 執行 fitness
fitness_value = fitness(order, leadtime, begin_inventory, bom, target_fill_rate)

# 輸出結果
print("Fitness Value:", fitness_value)

