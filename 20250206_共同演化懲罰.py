
import numpy as np
from scipy.stats import norm
np.set_printoptions(suppress=True, precision=6)
import time
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from functools import partial
from scipy.stats import skew, kurtosis, shapiro,poisson
from scipy.stats import kstest, norm


'''=================設定參數========================='''

num_products = 3
num_materials = 4
num_periods = 5  
leadtime = 1
inventory_cost_per_unit = 2
num_generations = 2
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
    # 設定初始樣本參數
    
    initial_sample_size = 50  # 初始樣本數
    max_sample_size = 2500   # 最大樣本數，防止無限循環
    epsilon = 0.01            # 收斂條件：均值與期望值的差距
    lam = 30                  # 泊松分布的參數（期望值）
    sigma_threshold = np.sqrt(lam)    # 收斂條件：標準差閾值
    normality_threshold = 0.05  # 常態性檢定 p-value 閾值

    sample_size = initial_sample_size
    converged = False



    while not converged:

        # 生成 LHS 樣本
        lhs_samples = latin_hypercube_sampling(lam,num_products * num_periods, sample_size)
        demand_samples = lhs_samples.reshape(sample_size, num_products, num_periods)

        # 計算樣本均值和標準差
        sample_mean = np.mean(demand_samples)
        sample_std = np.std(demand_samples)
        skewness =  skew(demand_samples)
        kurt = kurtosis(demand_samples)
        # 改用 Kolmogorov-Smirnov 檢驗
        ks_statistic, ks_p_value = kstest(demand_samples.flatten(), 'norm', args=(sample_mean, sample_std))

        print(f"樣本數: {sample_size}, 均值: {sample_mean:.2f}, 標準差: {sample_std:.2f}")
        print(f"Kolmogorov-Smirnov 檢驗 p-value: {ks_p_value:.4f}")



        # 檢查收斂條件
        if abs(sample_mean - lam) < epsilon and abs(sample_std - sigma_threshold) < epsilon and ks_p_value < normality_threshold:
            converged = True
        else:
            sample_size += 50  # 增加樣本數
            if sample_size > max_sample_size:
                print("樣本數達到上限，未能滿足收斂條件")
                break

    '----計算平均滿足率----'
    cumulative_fill_rate = []
    real_stock_list = []

    # 計算平均滿足率
    for demand in demand_samples:
        # 呼叫evaluator
        material_demand, real_stock, material_unclaim = evaluator(demand, order, leadtime)

        # 計算滿足率
        shortage_com = np.where(material_unclaim < 0, np.minimum(-material_unclaim, material_demand), 0)  # 計算零件缺貨量
        safe_material_demand = np.where(material_demand == 0, 1e-10, material_demand)  # 避免分母為零
        com_fill_rate = 1 - (shortage_com / safe_material_demand)  # 零件滿足率

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

    return cumulative_fill_rate_mean, real_stock_mean, sample_size


# 輸入訂單的染色體與世代數，輸出最佳解及其適應值
def fitness_calculator(individual,gen,sample_size_history,w1,w2):
    # 將染色體轉換為訂單
    order = np.array(individual).reshape(num_materials, num_periods+1)

    # 呼叫fitness
    start_time = time.time()
    final_fill_rate,real_stock_mean,sample_size = fitness(order)
    end_time = time.time()
    execution_time = end_time - start_time  # 計算執行時間
    print("Execution Time: ", execution_time)
    individual.fill_rate = final_fill_rate  # 保存滿足率

    # 記錄樣本數
    sample_size_history.append(sample_size)

    '----計算適應值----'
    # 計算存貨成本
    inventory_cost = np.sum(real_stock_mean) * inventory_cost_per_unit

    # 計算懲罰值
    penalty = 0
    if np.any(final_fill_rate < target_fill_rate):
        penalty_value=1000# 懲罰值       
        unmet_penalty = np.sum(np.maximum(0, target_fill_rate - final_fill_rate))
        penalty = (w1 * unmet_penalty) + (w2 * np.count_nonzero(final_fill_rate < target_fill_rate))


    # 計算適應值(存貨成本+懲罰值)
    fitness_value = inventory_cost + penalty 
    print("f: fitness_value: ", fitness_value)

    return fitness_value,



#評估懲罰權重
def evaluate_penalty(ind,gen,sample_size_history):
    w1, w2 = ind  # 取得該 P₂ 個體的懲罰權重
    population = toolbox.population(n=100)  # 產生 P₁ 族群
    
    for order in population:
        order.fitness.values = fitness_calculator(order, gen,sample_size_history,w1,w2)  # 讓 P₁ 依據 P₂ 計算適應值

    best_fitness = min([ind.fitness.values[0] for ind in population])  # 取最佳適應值
    return best_fitness,

def genetic_algorithm():
    '----設定----'
    sample_size_history = []  # 用於記錄每個基因的樣本數
    population = toolbox.population(n=1000)  # 設定初始個體數量（P₁）
    penalty_population = toolbox.penalty_population(n=10)  # 設定 P₂（懲罰權重族群）

    hall_of_fame = tools.HallOfFame(10)  # 設定保留最佳解個體數量
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("fill_rate", np.mean, axis=0)  # 計算平均滿足率
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    logbook = tools.Logbook()  
    logbook.header = ["gen", "min", "avg", "max"]

    stable_threshold = 15  # 若最佳解連續15代未變化則停止
    best_fitness_history = []  # 記錄每代最佳解適應值
    stable_count = 0  

    '----開始運算----'
    for gen in range(num_generations): 
        # 1. **先進化 P₂（懲罰權重）**
        for ind in penalty_population:
            ind.fitness.values = evaluate_penalty(ind,gen,sample_size_history)  # 計算 P₂ 內懲罰權重的適應值

        # **選擇當前最好的懲罰權重**
        best_penalty = tools.selBest(penalty_population, 1)[0]
        w1, w2 = best_penalty

        # 2. **進化 P₁（訂單族群），使用 P₂ 內最好的懲罰權重**
        for ind in population:
            ind.fitness.values = fitness_calculator(ind, gen, sample_size_history, w1, w2)

        # **進行 P₁ 的交配與突變**
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.6, mutpb=0.3)

        # **計算適應值**
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(lambda ind: fitness_calculator(ind, gen, sample_size_history, w1, w2), invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # **菁英選擇，保留最佳基因**
        population = toolbox.select(offspring, k=len(population))  # 選擇最佳基因
        hall_of_fame.update(population)  # 更新全局最佳解

        # **強制保留全局最小適應度**
        global_best = hall_of_fame[0]
        if global_best not in population:
            population[0] = global_best

        # **記錄此世代最佳適應度**
        best_fitness = hall_of_fame[0].fitness.values[0]
        if len(best_fitness_history) > 0 and best_fitness == best_fitness_history[-1]:
            stable_count += 1
        else:
            stable_count = 0  
        best_fitness_history.append(best_fitness)

        # **記錄統計數據**
        record = stats.compile(population)
        logbook.record(gen=gen, **record)
        print(logbook.stream)

        # **若 fill rate 滿足條件則停止**
        all_fill_rates_satisfied = all(
            np.all(ind.fill_rate >= target_fill_rate) for ind in population
        )
        if all_fill_rates_satisfied:
            print(f"Algorithm converged after {gen} generations. All fill rates satisfied.")
            break

        # **進化 P₂（懲罰權重族群）**
        offspring_penalty = algorithms.varAnd(penalty_population, toolbox, cxpb=0.5, mutpb=0.2)
        for ind in offspring_penalty:
           ind.fitness.values = evaluate_penalty(ind,gen, sample_size_history)
        penalty_population = toolbox.select(offspring_penalty, k=len(penalty_population))

    '----最佳解與適應值----'
    best_order = np.array(hall_of_fame[0]).reshape(num_materials, num_periods + 1)
    print("Best Solution (Order):\n", best_order)
    print("Best Solution (Fill Rate):\n", hall_of_fame[0].fill_rate)
    return hall_of_fame[0], hall_of_fame[0].fitness.values, logbook



# 輸入生成維度、樣本數，輸出LHS矩陣
def latin_hypercube_sampling(lam,dimensions, samples): 
    lhs_samples = np.zeros((samples, dimensions))

    for i in range(dimensions):
        cut = np.linspace(0, 1, samples + 1)  #切割成樣本數量的空間
        uniform_samples = np.random.uniform(cut[:-1], cut[1:], size=samples) #每個區間產生均勻隨機樣本
        np.random.shuffle(uniform_samples) #順序隨機打散
        # 利用泊松分布的 PPF (Percent-Point Function) 映射均勻樣本到泊松分布
        poisson_samples = poisson.ppf(uniform_samples, mu=lam).astype(int)
        
        # 將映射後的樣本存入 LHS 矩陣
        lhs_samples[:, i] = poisson_samples
    return lhs_samples


def combined_selection(population,k, elitism_size):
    # 保留菁英
    best_individuals = tools.selBest(population, elitism_size)
    # 全局最小適應度值個體強制保留
    global_best = tools.selBest(population, 1)[0]
    if global_best not in best_individuals:
        best_individuals[0] = global_best  # 確保全局最優被保留
    # 隨機選擇剩餘個體
    remaining_individuals = tools.selRandom(population, k - len(best_individuals))
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



#建立P2(懲罰權重族群)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("PenaltyIndividual", list, fitness=creator.FitnessMax)  # P₂：懲罰權重個體
#定義P2族群
toolbox.register("attr_penalty", np.random.uniform, 10, 100)  # w1, w2 的初始值範圍
toolbox.register("penalty_individual", tools.initRepeat, creator.PenaltyIndividual, toolbox.attr_penalty, n=2)
toolbox.register("penalty_population", tools.initRepeat, list, toolbox.penalty_individual)




'''=================開始運算========================='''


# 執行 GA 並繪圖
start_time_ga = time.time()
best_solution, best_fitness, logbook = genetic_algorithm()  # 確保返回 logbook
end_time_ga = time.time()


'''=================結果分析========================='''

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
