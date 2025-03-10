
import numpy as np
from scipy.stats import norm
np.set_printoptions(suppress=True, precision=6)
import time
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from functools import partial
 ##建立矩陣
def evaluator(demand, order, leadtime, bom):  # 輸入一組需求與訂購資料 輸出缺貨矩陣(缺貨時間,缺貨數量)     
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
    print("begin_inventory", begin_inventory)

    real_order = order[:, 1:]           # 剩餘列為訂單矩陣
    print("real_order", real_order)
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
    print("inventory", inventory)
    real_stock=np.where(inventory<0,0,inventory) #補貨後庫存
    print("real_stock", real_stock)
    #計算每期缺貨數量
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

    return material_demand,inventory,shortage,material_unclaim,product_shortage




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
    sample_size = 1
    min_demand, max_demand = 10, 50
    
    cumulative_fill_rate = []
    
    # 生成樣本
    lhs_samples = latin_hypercube_sampling(num_products * num_periods, sample_size)
    print("LHS samples:", lhs_samples)
    demand_samples = np.floor(lhs_samples * (max_demand - min_demand + 1) + min_demand).astype(int)
    print("Demand samples:", demand_samples)
    demand_samples = demand_samples.reshape(sample_size, num_products, num_periods)
    print("Reshape demand samples:", demand_samples)

    for demand in demand_samples:
        material_demand,inventory,shortage,material_unclaim,product_shortage = evaluator(demand, order, leadtime, bom)
        print("Material demand:", material_demand)
        print("Inventory:", inventory)
        print("Shortage:", shortage)    
        print("Material unclaim:", material_unclaim)
        print("Product shortage:", product_shortage)
        #print("Product shortage matrix:", product_shortage_matrix)

       


        shortage_com = np.where(material_unclaim < 0, np.minimum(-material_unclaim, material_demand), 0)
        print("Shortage com:", shortage_com)
        safe_material_demand = np.where(material_demand == 0, 1e-10, material_demand)  # 避免分母為零
        shortage_com_tr = shortage_com.transpose(2, 1, 0)
        print("Shortage com tr:", shortage_com_tr)
        com_fill_rate = 1 - (shortage_com / safe_material_demand)
        print("Com fill rate:", com_fill_rate)
        com_fill_rate_tr = com_fill_rate.transpose(2, 1, 0)
        print("Com fill rate tr:", com_fill_rate_tr)
        pro_fill_rate = np.prod(com_fill_rate_tr, axis=1)  # 產品滿足率
        print("Pro fill rate:", pro_fill_rate)
        cumulative_fill_rate.append(pro_fill_rate)
        print("Cumulative fill rate:", cumulative_fill_rate)

    # 計算累積平均產品滿足率
    cumulative_fill_rate_array = np.array(cumulative_fill_rate)
    print("Cumulative fill rate array:", cumulative_fill_rate_array)
    cumulative_mean = np.mean(cumulative_fill_rate_array, axis=0)
    print("Cumulative mean:", cumulative_mean)

    avg_demand = np.mean(demand_samples, axis=0)  # 平均需求
    print("Avg demand:", avg_demand)

    return cumulative_fill_rate_array, cumulative_mean, len(cumulative_fill_rate), avg_demand



def fitness_function(order):

    # 計算庫存成本
    inventory_cost = np.sum(order) * inventory_cost_per_unit
    print("inventory_cost:", inventory_cost)
    start_time = time.time()
    samples, final_fill_rate, total_sample, avg_demand= fitness(order,num_products,num_periods)
    end_time = time.time()
    execution_time = end_time - start_time  # 計算執行時間
    print("inventory_cost:", inventory_cost)
    #懲罰項
    penalty = 0
    if np.any(final_fill_rate < target_fill_rate):

        gen=1
        num_generations=1
        product_weights = np.array([3, 2, 1])  # 對應產品1、產品2、產品3的權重
        # 定義時間權重 (隨時間增加)
        time_weights = np.array([1,2,3,4,5])  # 對應第1到第5期的權重
        
        generation_rate=(gen / num_generations) #現在世代/總世代
        generation_factor= np.square(generation_rate) #世代因子
        weights=1
        #未滿足的懲罰
        unmet_fill_rate = np.maximum(0, target_fill_rate - final_fill_rate)
        weighted_penalty = np.sum(product_weights[:, None] * time_weights[None, :] * unmet_fill_rate)
        penalty_value=230
        penalty = generation_factor *penalty_value*weighted_penalty
        print("penalty:", penalty)

    fitness_value = inventory_cost + penalty 
    print("f: fitness_value: ", fitness_value)
    print(f"Generation {gen}, Avg Demand:\n{avg_demand}")
    print("final_fill_rate:")
    print(final_fill_rate)

    return fitness_value,samples, final_fill_rate, total_sample, avg_demand, execution_time, penalty



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
order = np.array([
    [60,40, 80, 30, 10, 5],
    [40,80, 90, 40, 15, 10],
    [50,60, 100, 25, 12, 10],
    [30,20, 100, 50, 20, 10]
])
target_fill_rate = 0.95
inventory_cost_per_unit = 2

fitness_value,samples, final_fill_rate, total_sample, avg_demand, execution_time, penalty= fitness_function(order)
print("execution_time:", execution_time)
print("penalty:", penalty)
print("平均需求:", avg_demand)
print("最終產品滿足率:", final_fill_rate)
print("總樣本數:", total_sample)

# 繪製每個產品的滿足率曲線
for product in range(num_products):
    plt.plot(np.arange(num_periods), final_fill_rate[product], label=f"Product {product+1}")
plt.xlabel("Period")
plt.ylabel("Product Fill Rate")
plt.title("Product Fill Rate Curve")
plt.legend()
plt.show()

# 繪製所有產品的平均滿足率曲線
average_fill_rate = np.mean(final_fill_rate, axis=0)  # 計算每個時間期的平均滿足率
plt.plot(np.arange(num_periods), average_fill_rate, label="Average Fill Rate")
plt.xlabel("Period")
plt.ylabel("Average Fill Rate")
plt.title("Average Product Fill Rate Curve")
plt.legend()
plt.show()

# 繪製平均需求柱狀圖
average_demand_per_product = np.mean(avg_demand, axis=1)  # 計算每個產品的平均需求
plt.bar(np.arange(num_products), average_demand_per_product)
plt.xlabel("Product")
plt.ylabel("Average Demand")
plt.title("Average Demand per Product")
plt.show()

