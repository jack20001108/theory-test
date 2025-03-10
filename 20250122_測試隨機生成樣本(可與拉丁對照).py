#20250114 改變初始也生成加菁英選擇改輪盤
import numpy as np
np.set_printoptions(suppress=True, precision=6)
import time
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from functools import partial
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.spatial import distance_matrix
import seaborn as sns

'''=================設定參數========================='''

num_products = 3
num_materials = 4
num_periods = 5  
leadtime = 1
inventory_cost_per_unit = 2
num_generations = 25
target_fill_rate = 0.95


bom = np.array([
    [1, 0, 1],
    [2, 1, 0],
    [0, 1, 1],
    [1, 1, 0]
])

'''==================主程式=========================='''

# 輸入一組需求與訂單，輸出缺貨情況(缺貨時間,缺貨數量) 
def evaluator(demand, order, leadtime):  
    num_product,num_period = demand.shape   # (產品數量,期數)
    inventory = np.zeros((num_materials, num_period))   # 庫存(M*T+1)

    '----計算庫存----'
    # 計算庫存要用到的數據
    begin_inventory = np.tile(order[:, 0][:, None], (1, num_periods))   # 初始庫存--> order第一列,展開成(M*T)
    
    bom_demand = bom[:, :, None] * demand   # 零件需求展開(N,M,T)
    material_demand = bom_demand.transpose(2,0,1)   # 每個產品的需求(T,N,M)
    cumulative_demand = np.cumsum(bom@demand,axis=1)    # 累積需求(N,T)
    
    real_order = order[:, 1:]   # 每期訂貨量--> order第二列到最後一列
    arrival_order=np.roll(real_order, shift=leadtime, axis=1)   # order往後推leadtime期到達
    arrival_order[:,:leadtime]=0    # 在前leadtime期沒有到貨
    cumulative_arrival=np.cumsum(arrival_order, axis=1)  # 累積到貨

    # 計算庫存
    inventory=begin_inventory+cumulative_arrival-cumulative_demand
    real_stock=np.where(inventory<0,0,inventory) # 將庫存<0的數值取0

    '----計算未分配需求----'
    # 計算未分配需求要用到的數據
    shifted_bom_demand = np.roll(bom_demand, shift=-1, axis=1)  # 右移一格
    shifted_bom_demand[:, -1, :] = 0  # 將最後一列補0
    material_reverse_cumulative_demand =np.cumsum(shifted_bom_demand[:, ::-1, :],axis=1)[:, ::-1, :] # 1.反轉時間軸累加 2.再反轉回來
    
    changed_inventory=inventory[:, :,None] 
    material_period_end_inventory=changed_inventory.transpose(0,2,1) # 使庫存與未分配需求形狀相符

    # 計算未分配零件數量
    unclaim_inventory = (material_period_end_inventory + material_reverse_cumulative_demand) 
    material_unclaim=unclaim_inventory.transpose(2,0,1) # 每個產品的未分配需求
        # product_shortage=unclaim_inventory.transpose(1,0,2) # 各產品所需的零件，在各期間是否缺貨 

    return material_demand,real_stock,material_unclaim

# 輸入一組訂單，輸出平均滿足率與存貨量
def fitness(order):
    '----生成隨機需求樣本----'
    # 設定參數
    sample_size = 100
    min_demand, max_demand = 10, 50
    lam=30

    # 生成樣本
    poi_samples = poisson_random_sampling(lam,num_products * num_periods, sample_size)
    
    demand_samples = poi_samples.reshape(sample_size, num_products, (num_periods)) # 轉成(樣本數,產品數,期數)

    # 分析 LHS 樣本的分布
    print("\n分析 LHS 樣本分布：")
    analyze_samples(demand_samples, "LHS")

    # 方法 1: 距離分布分析
    analyze_distance_distribution(demand_samples)

    # 方法 2: 最小距離比較 (只檢測 LHS)
    analyze_min_distance(demand_samples)

    # 方法 3: 覆蓋率檢查
    analyze_coverage(demand_samples, grid_size=10)

    '----計算平均滿足率----'
    cumulative_fill_rate = []
    real_stock_list = [] 

    # 計算平均滿足率
    for demand in demand_samples:
        # 呼叫evaluator
        material_demand,real_stock,material_unclaim= evaluator(demand, order, leadtime)
        
        # 計算滿足率
        shortage_com = np.where(material_unclaim < 0, np.minimum(-material_unclaim, material_demand), 0) # 計算零件缺貨量
        safe_material_demand = np.where(material_demand == 0, 1e-10, material_demand)  # 避免分母為零
        com_fill_rate = 1 - (shortage_com / safe_material_demand) # 零件滿足率

        com_fill_rate_tr = com_fill_rate.transpose(2, 1, 0)  # 轉置為(T,N,M)
        pro_fill_rate = np.min(com_fill_rate_tr, axis=1)  # 產品滿足率
        cumulative_fill_rate.append(pro_fill_rate)  # 累積產品滿足率

        # 計算存貨量
        real_stock_list.append(real_stock)

    # 計算累積平均產品滿足率
    cumulative_fill_rate_array = np.array(cumulative_fill_rate)
    cumulative_fill_rate_mean = np.mean(cumulative_fill_rate_array, axis=0)
    # 計算累積存貨量
    real_stock_array = np.array(real_stock_list)
    real_stock_mean = np.mean(real_stock_array, axis=0)

    return cumulative_fill_rate_mean,real_stock_mean



# 輸入訂單的染色體與世代數，輸出最佳解及其適應值
def fitness_calculator(individual,gen):
    # 將染色體轉換為訂單
    order = np.array(individual).reshape(num_materials, num_periods+1)

    # 呼叫fitness
    start_time = time.time()
    final_fill_rate,real_stock_mean = fitness(order)
    end_time = time.time()
    execution_time = end_time - start_time  # 計算執行時間
    print("Execution Time: ", execution_time)
    individual.fill_rate = final_fill_rate  # 保存滿足率

    '----計算適應值----'
    # 計算存貨成本
    inventory_cost = np.sum(real_stock_mean) * inventory_cost_per_unit

    # 計算懲罰值
    penalty = 0
    if np.any(final_fill_rate < target_fill_rate):
        penalty_value=1000# 懲罰值       
        #generation_rate= np.square(gen / num_generations)   # 世代權重        
        unmet_penalty =  np.sum(np.maximum(0, target_fill_rate - final_fill_rate)) # 未滿足差距
        #計算懲罰值        
        penalty =  penalty_value*unmet_penalty

    # 計算適應值(存貨成本+懲罰值)
    fitness_value = inventory_cost + penalty 
    print("f: fitness_value: ", fitness_value)

    return fitness_value,

 
def genetic_algorithm():
    '----設定----'
    population = toolbox.population(n=1000)  # 設定初始個體數量
    hall_of_fame = tools.HallOfFame(10)  # 設定保留最佳解個體數量
    stats = tools.Statistics(lambda ind: ind.fitness.values)  # 針對個體計算適應值並累計統計數據
    stats.register("fill_rate", np.mean, axis=0)  # 計算平均滿足率
    stats.register("min", np.min)  # 計算最小值
    stats.register("avg", np.mean)  # 計算平均值
    stats.register("max", np.max)  # 計算最大值
    # 紀錄運算過程
    logbook = tools.Logbook()  
    logbook.header = ["gen", "min", "avg", "max"] 

    '----收斂條件----'
    
    best_fitness_history = []  # 記錄每代最佳解適應值
    stable_count = 0  

    '----開始運算----'
    for gen in range(num_generations): 
        # 呼叫fitness_calculator，計算當前世代個體適應值
        toolbox.evaluate = partial(fitness_calculator, gen=gen)

        # 交配、突變
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.6, mutpb=0.3) 

        # 計算適應值
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 菁英選擇，紀錄最佳基因
        population = toolbox.select(offspring)  # 選擇最佳基因
        hall_of_fame.update(population)  # 更新全局最佳解

        # 強制保留全局最小適應度
        global_best = hall_of_fame[0]
        if global_best not in population:
            population[0] = global_best

        # 記錄此世代最佳適應度
        best_fitness = hall_of_fame[0].fitness.values[0]
        if len(best_fitness_history) > 0 and best_fitness == best_fitness_history[-1]:
            stable_count += 1  # 此世代的最佳適應度未改變+1
        else:
            stable_count = 0  # 若有改善，重新計算
        best_fitness_history.append(best_fitness)

        # 記錄統計數據
        record = stats.compile(population)
        logbook.record(gen=gen, **record)
        print(logbook.stream)

        # fill rate 滿足限制式條件
        all_fill_rates_satisfied = all(
            np.all(ind.fill_rate >= target_fill_rate) for ind in population
        )

        if all_fill_rates_satisfied:
            print(f"Algorithm converged after {gen} generations. All fill rates satisfied.")
            break



    '----最佳解與適應值----'
    best_order = np.array(hall_of_fame[0]).reshape(num_materials, num_periods + 1)
    print("Best Solution (Order):\n", best_order)
    print("Best Solution (Fill Rate):\n", hall_of_fame[0].fill_rate)
    return hall_of_fame[0], hall_of_fame[0].fitness.values, logbook


# 輸入生成維度、樣本數，輸出LHS矩陣
def latin_hypercube_sampling(dimensions, samples): 
    lhs_samples = np.zeros((samples, dimensions))
    for i in range(dimensions):
        cut = np.linspace(0, 1, samples + 1)  #切割成樣本數量的空間
        uniform_samples = np.random.uniform(cut[:-1], cut[1:], size=samples) #每個區間產生均勻隨機樣本
        np.random.shuffle(uniform_samples) #順序隨機打散
        lhs_samples[:, i] = uniform_samples 
    return lhs_samples

def poisson_random_sampling(lam, dimensions, samples):
    """
    隨機泊松抽樣函數
    lam: 泊松分布的參數 lambda
    dimensions: 維度數
    samples: 樣本數
    """
    # 使用泊松分布生成樣本
    poisson_samples = np.random.poisson(lam=lam, size=(samples, dimensions))
    return poisson_samples


def analyze_samples(demand_samples, method):
    # 將樣本展開為一維，方便統計
    flattened_samples = demand_samples.flatten()

    # 計算基本統計數據
    print(f"\n[{method}] 樣本統計分析：")
    print(f"最小值: {np.min(flattened_samples)}")
    print(f"最大值: {np.max(flattened_samples)}")
    print(f"平均值: {np.mean(flattened_samples):.2f}")
    print(f"標準差: {np.std(flattened_samples):.2f}")
    
    # 視覺化分佈
    plt.figure(figsize=(10, 6))
    plt.hist(flattened_samples, bins=30, alpha=0.7, color='blue', label=f'{method} Distribution')
    plt.xlabel("Demand Value")
    plt.ylabel("Frequency")
    plt.title(f"Demand Samples Distribution ({method})")
    plt.legend()
    plt.grid()
    plt.show()

def analyze_distance_distribution(lhs_samples):
    # 確保樣本為 2D 數組 (n_samples, n_features)
    reshaped_samples = lhs_samples.reshape(lhs_samples.shape[0], -1)  # 展開為 2D

    # 計算距離矩陣
    lhs_distances = distance_matrix(reshaped_samples, reshaped_samples)

    # 提取上三角的非零距離
    lhs_distances = lhs_distances[np.triu_indices_from(lhs_distances, k=1)]

    # 視覺化距離分布
    plt.figure(figsize=(10, 6))
    sns.histplot(lhs_distances, kde=True, color='blue', label='LHS', bins=30)
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.title("LHS Distance Distribution")
    plt.legend()
    plt.grid()
    plt.show()

def analyze_min_distance(lhs_samples):
    # 確保樣本為 2D 數組 (n_samples, n_features)
    reshaped_samples = lhs_samples.reshape(lhs_samples.shape[0], -1)  # 展開為 2D

    # 計算最小距離
    lhs_min_distance = np.min(distance_matrix(reshaped_samples, reshaped_samples)[np.triu_indices(len(reshaped_samples), k=1)])

    print(f"LHS 最小距離: {lhs_min_distance:.4f}")

def analyze_coverage(lhs_samples, grid_size):
    # 確保樣本點在 [0, 1] 範圍內
    lhs_samples = np.clip(lhs_samples, 0, 1)

    # 將樣本點映射到格子
    lhs_grid = (lhs_samples * grid_size).astype(int)

    # 計算覆蓋的格子數量
    unique_grids = np.unique(lhs_grid, axis=0)  # 唯一格子數
    total_grids = grid_size ** lhs_samples.shape[1]  # 總格子數
    lhs_coverage = len(unique_grids) / total_grids

    print(f"LHS 覆蓋率: {lhs_coverage:.4f}")
    return lhs_coverage






def combined_selection(population, elitism_size):
    # 保留菁英
    best_individuals = tools.selBest(population, elitism_size)
    # 全局最小適應度值個體強制保留
    global_best = tools.selBest(population, 1)[0]
    if global_best not in best_individuals:
        best_individuals[0] = global_best  # 確保全局最優被保留
    # 隨機選擇剩餘個體
    remaining_individuals = tools.selRandom(population, len(population) - elitism_size)
    return best_individuals + remaining_individuals







'''=================設定演算法設定========================='''

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  #適應值最小化
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_int", np.random.randint,0 ,200)  #每個基因的值（訂單 0 ~ 200）
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=num_materials * (num_periods+1))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", partial(fitness_calculator, gen=0))
toolbox.register("mate", tools.cxUniform, indpb=0.7)  #均勻交配
toolbox.register("mutate", tools.mutUniformInt, low=0, up=200, indpb=0.2)  #突變
#toolbox.register("select", tools.selRoulette)#選擇
#toolbox.register("select", tools.selBest)
#toolbox.register("select", tools.selTournament, tournsize=3) #競賽選擇
toolbox.register("select", combined_selection, elitism_size=10)  # 菁英挑選數量





'''=================開始運算========================='''


# 執行 GA 並比較 LHS 與 RS 的效果
#print("使用拉丁超立方抽樣 (LHS):")
#start_time_lhs = time.time()
#best_solution_lhs, best_fitness_lhs, logbook_lhs = genetic_algorithm()
#end_time_lhs = time.time()
order = np.array([
    [6,10, 20, 30, 10, 5],
    [8,15, 25, 20, 15, 10],
    [5,12, 18, 15, 12, 10],
    [12,8, 10, 15, 20, 10]
])
cumulative_fill_rate_mean,real_stock_mean=fitness(order)



'''=================結果分析========================='''

# 可視化 LHS 與 RS 的適應值分佈
#gen_lhs = logbook_lhs.select("gen")
#min_lhs = logbook_lhs.select("min")
#gen_rs = logbook_rs.select("gen")
#min_rs = logbook_rs.select("min")

#plt.figure(figsize=(10, 6))
#plt.plot(gen_lhs, min_lhs, label="LHS", marker='o', linestyle='-', linewidth=1.5)
#plt.plot(gen_rs, min_rs, label="RS", marker='s', linestyle='--', linewidth=1.5)
#plt.xlabel("Generation")
#plt.ylabel("Fitness")
#plt.title("LHS vs RS Fitness Convergence")
#plt.legend()
#plt.grid(True)
#plt.show()

