import numpy as np
import time

# "零件"庫存與缺貨計算
def evaluator(demand, order, leadtime, begin_inventory, bom):
    #矩陣形狀 (define metrix shape)
    num_simulations, num_products, num_periods = demand.shape #三維(模擬次數K,產品數量N,期數T)
    num_material = begin_inventory.shape[0] #零件數量 

    #初始庫存矩陣
    inventory = np.zeros((num_simulations, num_material, num_periods + 1))  # K*M*(T+1) 
    inventory[:, :, 0] = begin_inventory #初始庫存

    #計算零件需求
    component_demand = bom[:, :, None] * demand[:, None, :, :] # K*M*N*T
    #計算累積需求
    cumulate_demand = np.cumsum(component_demand.sum(axis=2), axis=2)  # K*M*T

    #計算累積到貨
    arrival_order = np.roll(order, shift=leadtime, axis=1)  #滾動補貨前置期
    arrival_order[:, :leadtime] = 0  #前 leadtime 期無到貨
    arrival_order = np.expand_dims(arrival_order, axis=0).repeat(num_simulations, axis=0)  # 擴展至K模擬情境
    cumulate_arrival = np.cumsum(arrival_order, axis=2)  # K*M*T

    #計算每期庫存
    inventory[:, :, 1:] = begin_inventory[None, :, None] + cumulate_arrival - cumulate_demand

    #計算缺貨
    shortage = np.where(inventory[:, :, 1:] < 0, -inventory[:, :, 1:], 0)  #該期缺貨數量

    #計算未分配需求
    shift_bom_demand=np.roll(component_demand,shift=-1,axis=1) 
    shift_bom_demand[:,-1,:,:]=0

    reverse_cumulative_demand = np.cumsum(component_demand[:, ::-1, :, :], axis=1)[:, ::-1, :, :]  #反累積需求

    reverse_inventory=inventory[:, :, 1:,None]
    reverse2_inventory=reverse_inventory.transpose(0,1,3,2)


    unclaim_inventory = (
        reverse2_inventory
        + reverse_cumulative_demand  #未分配庫存
    )

    reshape_unclaim=unclaim_inventory.transpose(0,3,1,2)
    aligned_bom_demand = component_demand.transpose(0,3,1,2)


    #計算零件滿足率
    shortage_com = np.where(
        reshape_unclaim < 0,
        np.minimum(-reshape_unclaim, aligned_bom_demand),
        0
    )

    # 避免分母為零
    safe_aligned_bom_demand = np.where(aligned_bom_demand == 0, 1e-10, aligned_bom_demand)

    # fill rate
    com_fill_rate = 1 - (shortage_com / safe_aligned_bom_demand)

    return inventory, shortage, unclaim_inventory, com_fill_rate



def fitness(order, leadtime, begin_inventory, bom, target_fill_rate, num_simulations=1000, demand_mean=15, demand_std=5, num_periods=5, num_products=3):
    start_time = time.time()
    #隨機產生1000組模擬需求
    demands = np.random.normal(
        loc=demand_mean,
        scale=demand_std,
        size=(num_simulations, num_products, num_periods)
    ).astype(int)
    demands = np.maximum(demands, 0)  #生成數字若為負
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
    print(f"Execution time for 1000 simulations: {execution_time:.4f} seconds")
    return fitness_value


#基因演算法

#初始母體(order)
def initialize_population(pop_size, num_material, num_periods, min_order, max_order,target_fill_rate):
    population = np.random.randint(
        low=min_order, high=max_order, size=(pop_size, num_material, num_periods)
    )

    return population

#計算fitness
def calculate_fitness(population, leadtime, begin_inventory, bom, target_fill_rate):
    fitness_values = []
    for order in population:
    
        average_product_fill_rate = fitness(
            order, leadtime, begin_inventory, bom, target_fill_rate
        )
        fitness_value=(order, leadtime, begin_inventory, bom, target_fill_rate)
        ################ 假設目標是最大化滿足率
        fitness_values.append(average_product_fill_rate)

    return np.array(fitness_values)


#挑選母代
def select_parents(population, fitness_values, num_parents):

    selected_indices = np.argsort(fitness_values)[-num_parents:]  # 取適應度最高的個體
    parents = population[selected_indices]

    return parents




#單點交配
def crossover(parents, num_offspring):
    offspring = []
    num_parents = parents.shape[0]  
    num_material, num_periods = parents.shape[1], parents.shape[2]  # 材料數量和時間期數

    for _ in range(num_offspring):
        parent_indices = np.random.choice(num_parents, size=2, replace=False)
        parent1, parent2 = parents[parent_indices[0]], parents[parent_indices[1]]

        # 單點交叉，確保交叉點有效
        crossover_point = np.random.randint(1, num_periods)  # 確保交叉點在有效範圍內
        child = np.hstack((parent1[:, :crossover_point], parent2[:, crossover_point:]))
        offspring.append(child)

    return np.array(offspring)



#突變
def mutate(offspring, mutation_rate, min_order, max_order):
    for child in offspring:
        if np.random.rand() < mutation_rate:
            i, j = np.random.randint(0, child.shape[0]), np.random.randint(0, child.shape[1])
            child[i, j] = np.random.randint(min_order, max_order)
    return offspring

#基因演算法主程式
def genetic_algorithm(
    leadtime, begin_inventory, bom, num_material, num_periods, target_fill_rate,
    pop_size=50, generations=100, num_parents=10, mutation_rate=0.1, 
    min_order=5, max_order=50
):
    # 初始化種群
    population = initialize_population(pop_size, num_material, num_periods, min_order, max_order,target_fill_rate)

    for generation in range(generations):
        # 計算適應度
        fitness_values = calculate_fitness(population, leadtime, begin_inventory, bom, target_fill_rate)

        # 選擇父代
        parents = select_parents(population, fitness_values, num_parents)
        
        # 交叉生成子代
        offspring = crossover(parents, pop_size - num_parents)
        
        # 子代變異
        offspring = mutate(offspring, mutation_rate, min_order, max_order)
        
        # 更新種群
        population[:num_parents] = parents
        population[num_parents:] = offspring
       
        # 紀錄最佳解
        best_fitness = np.max(fitness_values)
        print(best_fitness)

    
    # 返回最佳解
    best_index = np.argmax(fitness_values)


    return population[best_index], best_fitness



# 給定bom表
bom = np.array([
    [1, 0, 1],
    [2, 1, 0],
    [0, 1, 1],
    [1, 1, 0]
])

begin_inventory = np.array([50, 40, 30, 20])
leadtime = 1





target_fill_rate = np.array([0.95, 0.90, 0.85]) 
best_order, best_fitness = genetic_algorithm(
    leadtime=leadtime, 
    begin_inventory=begin_inventory, 
    bom=bom, 
    num_material=bom.shape[0],  # 零件數量 (根據 BOM)
    num_periods=5,  # 時間期數
    target_fill_rate=target_fill_rate,  # 傳入目標滿足率
    pop_size=50,  # 種群大小
    generations=100,  # 遺傳演算法代數
    num_parents=10,  # 父代數量
    mutation_rate=0.1,  # 變異率
    min_order=5,  # 最小補貨量
    max_order=50  # 最大補貨量
)

print("最佳 order 解：\n", best_order)
print("最佳適應度 (平均滿足率)：", best_fitness)
