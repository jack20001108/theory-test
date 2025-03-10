import numpy as np
import time
from randomgen import Xoroshiro128

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




def fitness(order, leadtime, begin_inventory, bom, target_fill_rate):
    start_time = time.time()

    demands = generate_multi_product_demand(num_simulations, num_products, num_periods, mean_demands, target_error)
    #呼叫evaluator
    inventory, shortage, unclaim_inventory, com_fill_rate = evaluator(demands, order, leadtime, begin_inventory, bom)
    

    #計算產品層級的平均滿足率
    product_fill_rate_by_period = np.prod(com_fill_rate, axis=2).mean(axis=0)  #每期平均滿足率
    average_product_fill_rate = np.mean(product_fill_rate_by_period, axis=0)
    

    diff=np.abs(average_product_fill_rate-target_fill_rate)
    #print("diff")
    #print(diff)
    weights = np.array([0.33, 0.33, 0.33])  # 三個產品的權重

    if weights is not None:
        diff=diff*weights

    #print("diff2")
    #print(diff)
    fitness_value=-np.sum(diff)


    #總平均滿足率
    #all_fill_rate=np.mean(average_product_fill_rate,axis=0)   
    end_time = time.time()
    execution_time = end_time - start_time
    #print(f"Fitness value: {fitness_value}")
    #print(f"Execution time for 1000 simulations: {execution_time:.4f} seconds")
    return fitness_value

# 動態泊松需求生成函數
def dynamic_poisson_simulation(mean_demand, target_error, num_simulations, max_samples=10000):
    rng = np.random.Generator(Xoroshiro128(seed=42))
    samples = []
    n = 100  # 初始樣本數

    while n <= max_samples:
        # 生成新樣本
        new_samples = rng.poisson(lam=mean_demand, size=n)
        samples.extend(new_samples)

        # 計算樣本均值和標準誤差
        sample_mean = np.mean(samples)
        sample_std_dev = np.std(samples, ddof=1)
        standard_error = 1.96 * sample_std_dev / np.sqrt(len(samples))

        #print(f"樣本數: {len(samples)}, 平均值: {sample_mean:.2f}, 誤差範圍: ±{standard_error:.2f}")

        # 判斷是否滿足目標誤差
        if standard_error <= target_error:
            break

        # 增加樣本數
        n *= 2  # 每次翻倍增加樣本數

    return np.random.choice(samples, size=num_simulations)  # 返回 num_simulations 個隨機樣本

# 多產品需求生成函數
def generate_multi_product_demand(num_simulations, num_products, num_periods, mean_demands, target_error):
    
    assert len(mean_demands) == num_products, "平均需求長度必須等於產品數量"
    
    demand_matrix = np.zeros((num_simulations, num_products, num_periods))
    
    for product in range(num_products):
        for period in range(num_periods):
            # 為每個產品的每個時間期生成 K 次模擬需求
            demand_values = dynamic_poisson_simulation(
                mean_demand=mean_demands[product],
                target_error=target_error,
                num_simulations=num_simulations
            )
            # 填充需求矩陣
            demand_matrix[:, product, period] = demand_values
        print(f"產品 {product+1} 完成")
        print(f"模擬需求：", demand_values)
        print(f"模擬需求矩陣：", demand_matrix[:, product, period])
    return demand_matrix.astype(int)

# 設定參數
num_simulations = 1000  # 模擬次數 K
num_products = 3       # 產品數量 N
num_periods = 5         # 時間期數 T
mean_demands = [10, 30, 55]  # 每個產品的平均需求 λ
target_error = 0.1      # 目標誤差

# 生成需求矩陣
demand_matrix = generate_multi_product_demand(num_simulations, num_products, num_periods, mean_demands, target_error)



# 測試參數
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


target_fill_rate = np.array([0.95, 0.90, 0.85]) 


fitness_value=fitness(order, leadtime, begin_inventory, bom, target_fill_rate)

# 輸出結果
print("生成的需求矩陣：")
print(demand_matrix)  # 顯示前兩組模擬的需求矩陣
print("生成的需求矩陣形狀：", demand_matrix.shape)  # 應為 (1000, 3, 5)