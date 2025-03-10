#20250114 改變初始也生成加菁英選擇改輪盤
import numpy as np
from scipy.stats import norm
np.set_printoptions(suppress=True, precision=6)
import time
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from functools import partial

'''==================evaluator======================'''
def evaluator(demand, order, leadtime, bom):  # 輸入一組需求與訂購資料 輸出缺貨矩陣(缺貨時間,缺貨數量) 
    ##建立矩陣
    
    num_product,num_period = demand.shape #產品數量,時間期數
    num_material=num_materials #零件數量
    bom_demand = bom[:, :, None] * demand  #零件需求展開(N,M,T)
    #print("bom_demand", bom_demand)
    inventory = np.zeros((num_material, num_period)) #庫存(M*T+1)
    #inventory[:,0] = begin_inventory #初始庫存
    shortage=np.zeros((num_material, num_period)) #缺貨(M*T)
    #計算庫存要用到的數據
    begin_inventory = order[:, 0]  # 第一列為初始庫存
    begin_inventory = begin_inventory[:, None]  # 變成 (4, 1)
    begin_inventory = np.tile(begin_inventory, (1, num_periods)) 

    real_order = order[:, 1:]           # 剩餘列為訂單矩陣
    material_demand = bom_demand.transpose(2,0,1) #每個產品的需求(T,N,M)
    #print("material_demand", material_demand)
    cumulative_demand = np.cumsum(bom@demand,axis=1) #累積需求
    #print("cumulative_demand", cumulative_demand)
    arrival_order=np.roll(real_order, shift=leadtime, axis=1) #在leadtime期補貨
    #print("arrival_order", arrival_order)
    arrival_order[:,:leadtime]=0 #在前leadtime期沒有補貨
    #print("arrival_order", arrival_order)
    cumulative_arrival=np.cumsum(arrival_order, axis=1) #累積到貨
    #print("cumulative_arrival", cumulative_arrival)


    #計算每期庫存
    inventory=begin_inventory+cumulative_arrival-cumulative_demand
    #print("inventory", inventory)
    real_stock=np.where(inventory<0,0,inventory) #補貨後庫存
    #紀錄缺貨數量
    cummulate_shortage=np.where(inventory<0,-inventory,0) #該期缺貨數量
    shortage[:,0]=cummulate_shortage[:, 0]
    shortage[:,1:]=cummulate_shortage[:, 1:] - cummulate_shortage[:, :-1]
    # 計算反累積需求 (全部倒過來累加)
    shifted_bom_demand = np.roll(bom_demand, shift=-1, axis=1)  #右移一格
    shifted_bom_demand[:, -1, :] = 0  #將最後一列補0
    reverse_cumulative_demand =np.cumsum(shifted_bom_demand[:, ::-1, :],axis=1)[:, ::-1, :] #反累積需求(時間軸轉回來)
    reverse_inventory=inventory[:, :,None] 
    reverse2_inventory=reverse_inventory.transpose(0,2,1) 
    #計算未分配零件數量
    unclaim_inventory = ( reverse2_inventory + reverse_cumulative_demand) #加入未分配需求 
    material_unclaim=unclaim_inventory.transpose(2,0,1)
    #產品缺貨矩陣
    product_shortage=unclaim_inventory.transpose(1,0,2) #每個產品，所需零件在各期間是否缺貨 

    return material_demand,inventory,shortage,material_unclaim,product_shortage,real_stock



def latin_hypercube_sampling(dimensions, samples): #輸入欲生成的維度、樣本數 輸出LHS矩陣
    result = np.zeros((samples, dimensions))
    for i in range(dimensions):
        cut = np.linspace(0, 1, samples + 1)  #切割成樣本數量的空間
        uniform_samples = np.random.uniform(cut[:-1], cut[1:], size=samples) #每個區間產生均勻隨機樣本
        np.random.shuffle(uniform_samples) #順序隨機打散
        result[:, i] = uniform_samples 
    return result

def fitness(order, num_products=3, num_periods=5):
    # 固定每次計算的樣本數
    sample_size = 100
    min_demand, max_demand = 10, 50
    
    cumulative_fill_rate = []
    
    # 生成樣本
    lhs_samples = latin_hypercube_sampling(num_products * num_periods, sample_size)

    #print("LHS samples:", lhs_samples)
    demand_samples = np.floor(lhs_samples * (max_demand - min_demand + 1) + min_demand).astype(int)
    #print("Demand samples:", demand_samples)
    demand_samples = demand_samples.reshape(sample_size, num_products, (num_periods))
    #print("Reshape demand samples:", demand_samples)

    for demand in demand_samples:
        material_demand, inventory, shortage, material_unclaim, product_shortage, real_stock = evaluator(demand, order, leadtime, bom)
        #print("Material demand:", material_demand)
        #print("Inventory:", inventory)
        #print("Shortage:", shortage)    
        #print("Material unclaim:", material_unclaim)
        #print("Product shortage:", product_shortage)

       


        shortage_com = np.where(material_unclaim < 0, np.minimum(-material_unclaim, material_demand), 0)
        #print("Shortage com:", shortage_com)
        safe_material_demand = np.where(material_demand == 0, 1e-10, material_demand)  # 避免分母為零
        shortage_com_tr = shortage_com.transpose(2, 1, 0)
        #print("Shortage com tr:", shortage_com_tr)
        com_fill_rate = 1 - (shortage_com / safe_material_demand)
        #print("Com fill rate:", com_fill_rate)
        com_fill_rate_tr = com_fill_rate.transpose(2, 1, 0)
        #print("Com fill rate tr:", com_fill_rate_tr)
        pro_fill_rate = np.prod(com_fill_rate_tr, axis=1)  # 產品滿足率
        #print("Pro fill rate:", pro_fill_rate)
        cumulative_fill_rate.append(pro_fill_rate)
        #print("Cumulative fill rate:", cumulative_fill_rate)

    # 計算累積平均產品滿足率
    cumulative_fill_rate_array = np.array(cumulative_fill_rate)
    #print("Cumulative fill rate array:", cumulative_fill_rate_array)
    cumulative_mean = np.mean(cumulative_fill_rate_array, axis=0)
    #print("Cumulative mean:", cumulative_mean)

    avg_demand = np.mean(demand_samples, axis=0)  # 平均需求
    #print("Avg demand:", avg_demand)

    return cumulative_fill_rate_array, cumulative_mean, len(cumulative_fill_rate), avg_demand,real_stock



def fitness_function(individual,gen,num_generations):

    order = np.array(individual).reshape(num_materials, num_periods+1)
    # 計算庫存成本
    
    #print("inventory_cost:", inventory_cost)
    start_time = time.time()
    samples, final_fill_rate, total_sample, avg_demand,real_stock = fitness(order,num_products,num_periods)
    end_time = time.time()
    execution_time = end_time - start_time  # 計算執行時間
    inventory_cost = np.sum(real_stock) * inventory_cost_per_unit
    #print("inventory_cost:", inventory_cost)
    #懲罰項
    penalty = 0
    if np.any(final_fill_rate < target_fill_rate):

        #gen=1
        #num_generations=1
        #product_weights = np.array([3, 2, 1])  # 對應產品1、產品2、產品3的權重
        # 定義時間權重 (隨時間增加)
        #time_weights = np.array([1,2,3,4,5])  # 對應第1到第5期的權重
        #beta = 0.9  # 折扣率
        # 計算每期的權重 w_t
        #period_weights = np.array([beta**(num_periods - t + 1) for t in range(1, num_periods + 1)])
        generation_rate=(gen / num_generations) #現在世代/總世代
        generation_factor= np.square(generation_rate) #世代因子
        weights=1
        #未滿足的懲罰
        unmet_fill_rate = np.maximum(0, target_fill_rate - final_fill_rate)
        weighted_penalty = np.sum(unmet_fill_rate)
        
        
        #weighted_penalty = np.sum(product_weights[:, None] * time_weights[None, :] * unmet_fill_rate)
        penalty_value=200
        penalty = generation_factor *penalty_value*weighted_penalty
        #print("penalty:", penalty)


    fitness_value = inventory_cost + penalty 
    print("f: fitness_value: ", fitness_value)
    print(f"Generation {gen}")
    #print("final_fill_rate:")
    #print(final_fill_rate)
    individual.fill_rate = final_fill_rate  # 保存滿足率
    individual.penalty = penalty  # 保存懲罰


    return fitness_value,


#def inverse_fitness(ind):
 #   return 1 / (1 + ind.fitness.values[0])  # 確保適應值越小，值越大
def combined_selection(population, elitism_size):
    offspring_size = len(population)  # 後代總數等於當前種群的大小
    best_individuals = tools.selBest(population, elitism_size)  # 保留菁英個體
    
    # 將適應值轉換為反比權重（越小越好）
    fitness_values = np.array([ind.fitness.values[0] for ind in population])
    inverse_fitness = 1 / (fitness_values + 1e-10)  # 避免除以 0
    weights = inverse_fitness / np.sum(inverse_fitness)  # 標準化為機率

    # 剩餘部分用自定義的輪盤選擇
    remaining_size = len(population) - elitism_size
    chosen_indices = np.random.choice(len(population), size=remaining_size, p=weights, replace=True)
    remaining_individuals = [population[i] for i in chosen_indices]

    # 返回菁英和輪盤選擇的結合
    return best_individuals + remaining_individuals

    







# 修改 GA 執行函數，確保返回 logbook（記錄物件）
def run_ga():
    population = toolbox.population(n=100)
    num_generations = 10
    hall_of_fame = tools.HallOfFame(5)  # 設定保留最優個體數量
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    #stats.register("avg", np.mean)
    #stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "min"]

    for gen in range(num_generations):
        toolbox.evaluate = partial(fitness_function, gen=gen, num_generations=num_generations)

        # 執行交配和突變
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.6, mutpb=0.3)

        # 評估子代
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # **菁英選擇：保留最優個體**
        population = toolbox.select(offspring)  # 選擇下一代個體，包含菁英和輪盤選擇
        hall_of_fame.update(population)

        # 記錄統計數據
        record = stats.compile(population)
        logbook.record(gen=gen, **record)
        print(logbook.stream)

    best_order = np.array(hall_of_fame[0]).reshape(num_materials, num_periods + 1)
    print("Best Solution (Order):\n", best_order)
    print("Best Solution (Fill Rate):\n", hall_of_fame[0].fill_rate)
    return hall_of_fame[0], hall_of_fame[0].fitness.values, logbook


# 設定參數
num_products = 3
num_materials = 4
num_periods = 5  
leadtime = 1

bom = np.array([
    [1, 0, 1],
    [2, 1, 0],
    [0, 1, 1],
    [1, 1, 0]
])

target_fill_rate = 0.95
inventory_cost_per_unit = 2

#基因演算法設定
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  #最小化問題
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_int", np.random.randint,0 ,200)  #每個基因的值（訂單 0 ~ 200）
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=num_materials * (num_periods+1))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", partial(fitness_function, gen=0))
toolbox.register("mate", tools.cxUniform, indpb=0.7)  #均勻交配
toolbox.register("mutate", tools.mutUniformInt, low=0, up=200, indpb=0.2)  #突變
#toolbox.register("select", tools.selRoulette)#選擇
#toolbox.register("select", tools.selBest)
toolbox.register("select", combined_selection, elitism_size=10)  # 增加菁英數量





# 執行 GA 並繪圖
start_time = time.time()
best_solution, best_fitness, logbook = run_ga()  # 確保返回 logbook
end_time = time.time()

# 提取收斂數據並繪圖
gen = logbook.select("gen")  # 從 logbook 提取各代數據
avg = logbook.select("avg")
min_ = logbook.select("min")
max_ = logbook.select("max")

plt.figure(figsize=(10, 6))
#plt.plot(gen, avg, label="AVG fitness", marker='o', linestyle='-', linewidth=1.5)
plt.plot(gen, min_, label="MIN fitness", marker='s', linestyle='--', linewidth=1.5)
#plt.plot(gen, max_, label="MAX fitness", marker='^', linestyle=':', linewidth=1.5)
plt.xlabel("generation", fontsize=12)
plt.ylabel("fitness", fontsize=12)
plt.title("GA convergence", fontsize=14)
plt.legend(loc="best", fontsize=10)
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(gen, avg, label="AVG fitness", marker='o', linestyle='-', linewidth=1.5)
#plt.plot(gen, min_, label="MIN fitness", marker='s', linestyle='--', linewidth=1.5)
plt.plot(gen, max_, label="MAX fitness", marker='^', linestyle=':', linewidth=1.5)
plt.xlabel("generation", fontsize=12)
plt.ylabel("fitness", fontsize=12)
plt.title("GA convergence", fontsize=14)
plt.legend(loc="best", fontsize=10)
plt.grid(True)
plt.show()
