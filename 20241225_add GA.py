import numpy as np

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
    #print("cumulate_demand：", cumulate_demand)

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

    # 計算 fill rate
    com_fill_rate = 1 - (shortage_com / safe_aligned_bom_demand)

    return inventory, shortage, unclaim_inventory, com_fill_rate



def fitness(order, leadtime, begin_inventory, bom, num_simulations=1000, demand_mean=15, demand_std=5, num_periods=5, num_products=3):

    # 隨機生成需求情境 (模擬所有情境)
    demands = np.random.normal(
        loc=demand_mean,
        scale=demand_std,
        size=(num_simulations, num_products, num_periods)
    ).astype(int)
    demands = np.maximum(demands, 0)  # 確保需求為非負數

    # 呼叫 evaluator 函數
    inventory, shortage, unclaim_inventory, com_fill_rate = evaluator(demands, order, leadtime, begin_inventory, bom)

    # 平均庫存與缺貨
    avg_inventory = inventory.mean(axis=0)  # 平均庫存
    avg_shortage = shortage.mean(axis=0)  # 平均缺貨

    # 計算產品層級的平均滿足率
    product_fill_rate_by_period = np.prod(com_fill_rate, axis=2).mean(axis=0)  # 每期平均滿足率
    average_product_fill_rate = np.mean(product_fill_rate_by_period, axis=0)  # 總平均滿足率
    all_fill_rate=np.mean(average_product_fill_rate,axis=0)    
    #print(inventory.shape)
    #return avg_inventory, avg_shortage, com_fill_rate, product_fill_rate_by_period, average_product_fill_rate
    return all_fill_rate


# 測試參數
bom = np.array([
    [1, 0, 1],
    [2, 1, 0],
    [0, 1, 1],
    [1, 1, 0]
])
#order = np.array([
#    [10, 20, 30, 10, 5],
#    [15, 25, 20, 15, 10],
#    [12, 18, 15, 12, 10],
#    [8, 10, 15, 20, 10]
#])
begin_inventory = np.array([50, 40, 30, 20])
leadtime = 1

# 執行 fitness
#avg_inventory, avg_shortage, com_fill_rate, product_fill_rate_by_period, average_product_fill_rate = fitness(
#    order, leadtime, begin_inventory, bom
#)

#print("平均庫存狀態：", avg_inventory)
#print("平均缺貨數量：", avg_shortage)
#print("\n零件滿足率矩陣:")
#print(com_fill_rate)
#print("每期每產品的平均滿足率：", product_fill_rate_by_period)
#print("每產品的平均滿足率：", average_product_fill_rate)
#print(com_fill_rate.shape)



def initialize_population(pop_size, num_material, num_periods, min_order, max_order):
    """
    初始化種群，產生多個 order 矩陣。
    """
    population = np.random.randint(
        low=min_order, high=max_order, size=(pop_size, num_material, num_periods)
    )

    return population


def calculate_fitness(population, leadtime, begin_inventory, bom):
    """
    計算每個個體的適應度。
    """
    fitness_values = []
    for order in population:
        all_fill_rate = fitness(
            order, leadtime, begin_inventory, bom
        )
        # 假設目標是最大化滿足率
        fitness_values.append(all_fill_rate)

    return np.array(fitness_values)



def select_parents(population, fitness_values, num_parents):
    """
    根據適應度選擇父代。
    """
    # 確保選出的父代形狀正確
    selected_indices = np.argsort(fitness_values)[-num_parents:]  # 取適應度最高的個體
    parents = population[selected_indices]

    return parents





def crossover(parents, num_offspring):
    """
    單點交叉操作，生成子代。
    """
    offspring = []
    num_parents = parents.shape[0]  # 父代數量
    num_material, num_periods = parents.shape[1], parents.shape[2]  # 材料數量和時間期數

    for _ in range(num_offspring):
        # 隨機選擇兩個父代的索引
        parent_indices = np.random.choice(num_parents, size=2, replace=False)
        parent1, parent2 = parents[parent_indices[0]], parents[parent_indices[1]]

        # 單點交叉，確保交叉點有效
        crossover_point = np.random.randint(1, num_periods)  # 確保交叉點在有效範圍內
        child = np.hstack((parent1[:, :crossover_point], parent2[:, crossover_point:]))
        offspring.append(child)

    return np.array(offspring)




def mutate(offspring, mutation_rate, min_order, max_order):
    """
    隨機變異操作。
    """
    for child in offspring:
        if np.random.rand() < mutation_rate:
            i, j = np.random.randint(0, child.shape[0]), np.random.randint(0, child.shape[1])
            child[i, j] = np.random.randint(min_order, max_order)
    return offspring


def genetic_algorithm(
    leadtime, begin_inventory, bom, num_material, num_periods, 
    pop_size=50, generations=100, num_parents=10, mutation_rate=0.1, 
    min_order=5, max_order=50
):
    # 初始化種群
    population = initialize_population(pop_size, num_material, num_periods, min_order, max_order)

    for generation in range(generations):
        # 計算適應度
        fitness_values = calculate_fitness(population, leadtime, begin_inventory, bom)

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


best_order, best_fitness = genetic_algorithm(
    leadtime=leadtime, 
    begin_inventory=begin_inventory, 
    bom=bom, 
    num_material=bom.shape[0],  # 零件數量 (根據 BOM)
    num_periods=5,  # 時間期數
    pop_size=50,  # 種群大小
    generations=100,  # 遺傳演算法代數
    num_parents=10,  # 父代數量
    mutation_rate=0.1,  # 變異率
    min_order=5,  # 最小補貨量
    max_order=50  # 最大補貨量
)

print("最佳 order 解：\n", best_order)
print("最佳適應度 (平均滿足率)：", best_fitness)
